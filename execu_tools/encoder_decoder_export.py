

@dataclass
class TorchExportableEncoderDecoderConfig:
    max_batch_size: int
    max_encoder_seq_len: int
    max_decoder_seq_len: int
    dtype: torch.dtype


class EncoderDecoderExportable(torch.nn.Module):
    """
    A wrapper module designed to make a `PreTrainedModel` exportable with `torch.export`,
    specifically for use with static caching. This module ensures that the exported model
    is compatible with further lowering and execution in `ExecuTorch`.

    Note:
        This class is specifically designed to support export process using `torch.export`
        in a way that ensures the model can be further lowered and run efficiently in `ExecuTorch`.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        exportable_config: TorchExportableEncoderDecoderConfig,
    ):
        """
        """
        super().__init__()

        # Sanity checks
        if model.generation_config is None:
            raise AssertionError(
                "The model must have a generation config to be exported with static caching. "
                "Please set `generation_config`."
            )

        if not model.config.is_encoder_decoder:
            raise AssertionError(
                "The model must be an encoder-decoder model to be exported with TorchExportableEncoderDecoder."
            )

        # verify that the model supports static caching:


        self.model = model
        self.config = exportable_config

        # things automagically pulled from the model:
        self.is_causal = True
        encoder_embedding_dim = self.model.config.d_model

        # Create the cache storage:
        encoder_cache = StaticCache(
            model.config,
            max_cache_len=self.config.max_batch_size,
            max_batch_size=self.config.max_encoder_seq_len,
            dtype=self.config.dtype,
        )
        decoder_cache = StaticCache(
            model.config,
            max_cache_len=self.config.max_batch_size,
            max_batch_size=self.config.max_encoder_seq_len,
            dtype=self.config.dtype,
        )
        self.static_cache = EncoderDecoderCache(decoder_cache, encoder_cache)

        if self.is_causal:
            causal_mask = torch.tril(
                torch.ones(
                    decoder_cache.max_cache_len,
                    decoder_cache.max_cache_len,
                    dtype=torch.bool,
                )
            )
            self.register_buffer("mask", causal_mask, persistent=False)

        encoder_seq_len = torch.tensor(0, dtype=torch.int32)
        self.register_buffer(
            "encoder_seq_len",
            encoder_seq_len,
            persistent=True,
        )
        # create a buffer for encoder output (bz,seq_len,embedding_dim):
        encoder_last_hidden_state = torch.zeros(
            self.config.max_batch_size,
            self.config.max_encoder_seq_len,
            encoder_embedding_dim,
            dtype=self.model.get_decoder().dtype,
        )
        self.register_buffer(
            "encoder_last_hidden_state", encoder_last_hidden_state, persistent=True
        )
        # create a buffer for encoder attention mask (bz,seq_len,embedding_dim):
        encoder_attention_mask = torch.zeros(
            self.config.max_batch_size,
            self.config.max_encoder_seq_len,
            encoder_embedding_dim,
            dtype=self.model.get_decoder().dtype,
        )
        self.register_buffer(
            "encoder_attention_mask", encoder_attention_mask, persistent=True
        )

    def forward_encoder(
        self,
        decoder_input_ids: torch.Tensor,
        decoder_cache_position: torch.Tensor,
        encoder_input_ids: torch.Tensor = None,
        encoder_attention_mask=None,
        encoder_sequence_length: torch.Tensor = None,
    ):
        """
        Forward pass of the module, which is compatible with the ExecuTorch runtime.

        Args:
            input_ids (`torch.Tensor`): Tensor representing current input token id to the module.
            cache_position (`torch.Tensor`): Tensor representing current input position in the cache.

        Returns:
            torch.Tensor: Logits output from the model.

        This forward adapter serves two primary purposes:

        1. **Making the Model `torch.export`-Compatible**:
            The adapter hides unsupported objects, such as the `Cache`, from the graph inputs and outputs,
            enabling the model to be exportable using `torch.export` without encountering issues.

        2. **Ensuring Compatibility with `ExecuTorch` runtime**:
            The adapter matches the model's forward signature with that in `executorch/extension/llm/runner`,
            ensuring that the exported model can be executed in `ExecuTorch` out-of-the-box.
        """

        # two paths in forward, either we update decoder input_ids and regenerate cross attention cache or we dont, and use previous generated values.
        if encoder_input_ids is None and encoder_sequence_length is None:
            # we need either encoder input ids or encoder sequence length to generate the encoder output.
            raise ValueError("encoder_input_ids or encoder_sequence_length must be provided")

        if encoder_input_ids is not None:
            # generate the encoder input ids:
            encoder = self.model.get_encoder()
            encoder_output = encoder(
                input_ids=encoder_input_ids, attention_mask=encoder_attention_mask
            )
            batch_size, seq_len, embed_dim = encoder_output.last_hidden_state.shape
            self.encoder_seq_len.fill_(seq_len)
            self.encoder_last_hidden_state[:batch_size, :seq_len, :] = (
                encoder_output.last_hidden_state
            )
            self.encoder_attention_mask[:batch_size, :seq_len, :] = (
                encoder_attention_mask
            )
            pass

        curr_batch_size, seqlen = decoder_input_ids.shape
        attn_mask = (
            self.mask[decoder_cache_position, :seqlen] if self.is_causal else None
        )

        # do assertions on the size of the last hidden state:

        # decoders take a base model output object
        encoder_output = BaseModelOutput(
            self.encoder_last_hidden_state[:curr_batch_size, : self.encoder_seq_len, :],
            None,
            None,
        )

        # assert that

        outs = self.model(
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=attn_mask,
            cache_position=decoder_cache_position,
            encoder_outputs=encoder_output,
            attention_mask=encoder_attention_mask,
            # encoder_attention_mask = self.encoder_attention_mask,
            past_key_values=self.static_cache,
            use_cache=True,
        )
        # outs = self.model(
        #     input_ids=input_ids,
        #     attention_mask=attn_mask,

        #     cache_position=cache_position,
        #     past_key_values=self.static_cache,
        #     use_cache=True,

        #     **kwargs,
        # )
        return outs.logits

    def forward_decoder(
        self,
        decoder_input_ids: torch.Tensor,
        decoder_cache_position: torch.Tensor,
        # parameters that should be auto provided:
        encoder_sequence_length: torch.Tensor = None,
    ):
        """
        Forward pass of the module, which is compatible with the ExecuTorch runtime.
        """