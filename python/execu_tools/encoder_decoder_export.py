import torch
from transformers.modeling_outputs import BaseModelOutput
from transformers.cache_utils import EncoderDecoderCache
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList
from transformers.tokenization_utils_fast import TokenizerFast
from transformers.convert_slow_tokenizer import convert_slow_tokenizer



class EncoderDecoderWrapper(torch.nn.Module):
    def __init__(self, model, cache):
        super().__init__()
        self.model = model

        self.shared_fqn = []
        # add the cache:
        self.cache = cache
        self.shared_fqn.append("cache")
        if type(self.cache) is EncoderDecoderCache:
            self_attn_cache = self.cache.self_attention_cache
            cross_attn_cache = self.cache.cross_attention_cache

            max_encoder_sequence_length = cross_attn_cache.key_cache[0].shape[2]
            _max_encoder_batch_size = cross_attn_cache.key_cache[0].shape[0]
            max_decoder_sequence_length = self_attn_cache.key_cache[0].shape[2]
            _max_decoder_batch_size = self_attn_cache.key_cache[0].shape[0]
            assert _max_encoder_batch_size == _max_decoder_batch_size
            max_batch_size = _max_decoder_batch_size
        else:
            raise ValueError("Cache is not an EncoderDecoderCache")

        # TODO: embedding dim should be dynamic based on model config.
        self.register_buffer(
            "encoder_output",
            torch.zeros(max_batch_size, max_encoder_sequence_length, 512),
        )

        self.shared_fqn.append("encoder_output")

        # setup static things that are not part of execution (follow generate roughly)
        # get the bos id
        self.generation_config, self.model_kwargs = (
            self.model._prepare_generation_config(
                self.model.generation_config, past_key_values=self.cache
            )
        )
        self.model._prepare_special_tokens(self.generation_config, False, None)

        # get the stopping criteria:
        stopping_criteria = StoppingCriteriaList()  # can override if needed
        self.prepared_stopping_criteria = self.model._get_stopping_criteria(
            generation_config=self.generation_config,
            stopping_criteria=stopping_criteria,
        )
        self.has_eos_stopping_criteria = any(
            hasattr(criteria, "eos_token_id")
            for criteria in self.prepared_stopping_criteria
        )

        # register a buffer to manage the unfinished sequences:
        # number of sequences is at mac the cache batch dimension:
        self.register_buffer(
            "unfinished_sequences", torch.ones(max_batch_size, dtype=torch.long)
        )
        self.shared_fqn.append("unfinished_sequences")

        self.register_buffer("cache_position", torch.zeros((1,), dtype=torch.long))
        self.shared_fqn.append("cache_position")

        self.register_buffer(
            "next_tokens", torch.zeros((max_batch_size, 1), dtype=torch.long)
        )  # should be 2d, with batch size as first dim.
        self.shared_fqn.append("next_tokens")
        # get the logits processors:
        logits_processor = LogitsProcessorList()  # can override if needed
        self.prepared_logits_processor = self.model._get_logits_processor(
            generation_config=self.generation_config,
            input_ids_seq_length=None,  # maybe needed?
            encoder_input_ids=None,  # maybe needed?
            prefix_allowed_tokens_fn=None,  # override if needed
            logits_processor=logits_processor,
            device=None,  # override if needed
            model_kwargs=self.model_kwargs,
            negative_prompt_ids=None,  # override if needed
            negative_prompt_attention_mask=None,  # override if needed
        )

        self.is_causal = True  # are they all causal?

        self.register_buffer(
            "decoder_attention_mask",
            torch.tril(
                torch.ones(
                    max_decoder_sequence_length,
                    max_decoder_sequence_length,
                    dtype=torch.bool,
                )
            ),
        )
        self.shared_fqn.append("decoder_attention_mask")
    def format_prompt(self, prompt=None):
        # idea is to format the prompt in a standard way, TODO: reevaluate this.
        # Marian models just have start of string token, so we just return that.
        return self.generation_config._decoder_start_token_tensor.unsqueeze(0)

    def get_shared_fqn(self):
        return self.shared_fqn

    def get_tokenizer_json(self):
        return self.tokenizer.to_json_string()

    def _process_next_tokens(
        self, batch_size, cache_position, next_token_scores, prev_decoder_outputs
    ):
        # greedy search
        next_tokens = torch.argmax(next_token_scores, dim=-1)

        # Create new decoder outputs by concatenating previous outputs with next tokens
        decoder_outputs = torch.cat(
            [prev_decoder_outputs, next_tokens.unsqueeze(1)], dim=1
        )

        # handle stopping criteria
        if self.has_eos_stopping_criteria:
            next_tokens = (
                next_tokens * self.unfinished_sequences[:batch_size]
                + self.generation_config._pad_token_tensor
                * (1 - self.unfinished_sequences[:batch_size])
            )[:batch_size]

        self.unfinished_sequences[:batch_size] = self.unfinished_sequences[
            :batch_size
        ] & ~self.prepared_stopping_criteria(decoder_outputs, next_token_scores)
        finished = self.unfinished_sequences.max() == 0

        # return None, None, None
        return finished, next_tokens.unsqueeze(1), decoder_outputs

    def _process_logits(self, prev_decoder_outputs, next_token_logits):
        """Process the logits from the decoder output using the prepared logits processor."""
        # get the next token logits and process them
        next_token_scores = self.prepared_logits_processor(
            prev_decoder_outputs, next_token_logits
        )
        return next_token_scores

    def run_decoder(
        self,
        encoder_outputs,
        encoder_attention_mask,
        decoder_inputs,
        decoder_attention_mask,
        cache_position: torch.Tensor,
        prev_decoder_outputs,
    ) -> dict:
        batch_size, seqlen = decoder_inputs.shape

        # since we are using kv-cache our attention mask should always be true?
        # TODO: check if this is correct.

        decoder_output = self.model(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_inputs,
            decoder_attention_mask=decoder_attention_mask,
            cache_position=cache_position,
            past_key_values=self.cache,
            return_dict=True,
        )

        # Process logits using the new helper function
        next_token_logits = decoder_output.logits[:, -1, :].clone().float()
        next_token_scores = self._process_logits(
            prev_decoder_outputs, next_token_logits
        )

        finished, next_tokens, decoder_outputs = self._process_next_tokens(
            batch_size, cache_position, next_token_scores, prev_decoder_outputs
        )
        # return None, None, None
        return finished, next_tokens, decoder_outputs

    def reset_encode_prefill(
        self,
        encoder_inputs: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        prefill_prompt: torch.Tensor,
    ):
        batch_size, encoder_sequence_length = encoder_inputs.shape

        if encoder_sequence_length > self.encoder_output.shape[1]:
            raise ValueError(
                "Encoder sequence length is greater than the max encoder sequence length"
            )
        if batch_size > self.encoder_output.shape[0]:
            raise ValueError("Batch size is greater than the max batch size")

        if len(prefill_prompt.shape) != 1:
            raise ValueError(
                "Prefill prompt must be a 1D tensor (prompt must be the same for all items in the batch)"
            )

        # reset the encoder output:
        self.encoder_output.fill_(0)

        # # reset the cache:
        self.cache.reset()
        self.cache_position.fill_(0)
        self.next_tokens.fill_(0)

        # # elements outside of the batch size are set to 0: (may not be needed)
        self.unfinished_sequences.fill_(0)
        narrowed_unfinished_sequences = torch.narrow(
            self.unfinished_sequences, 0, 0, batch_size
        )
        narrowed_unfinished_sequences.fill_(1)

        encoder = self.model.get_encoder()
        encoder_dict = encoder(
            input_ids=encoder_inputs,
            attention_mask=encoder_attention_mask,
        )

        narrowed_encoder_output = torch.narrow(
            self.encoder_output, 0, 0, batch_size
        ).narrow(1, 0, encoder_sequence_length)
        narrowed_encoder_output = encoder_dict["last_hidden_state"]

        encoder_model_output = BaseModelOutput(
            last_hidden_state=narrowed_encoder_output
        )

        # We are doing prefill:
        prefill_len = prefill_prompt.shape[0]
        prefill_positions = (
            torch.ones_like(prefill_prompt, dtype=torch.int64).cumsum(0) - 1
        )
        decoder_inputs = prefill_prompt.unsqueeze(0).repeat(batch_size, 1)

        # Initialize past_decoder_outputs for the first token
        past_decoder_outputs = decoder_inputs

        if self.is_causal:
            # TODO: look at this to see if it is needed. (in this particular case, probably not.)
            # in the general case, we need to use the cache position to find the
            # current position in the mask, however in this particular case we
            # know that tthe cache position is sequential and starts at zero,
            # so the shape of cache position-1 is the same as cache position.
            # start_pos = cache_position[-1].item()
            # torch._check_is_size(start_pos)
            # torch._check(start_pos < self.decoder_attention_mask.shape[0])
            _attn_row = self.decoder_attention_mask.select(
                0, prefill_positions.shape[0] - 1
            )
            decoder_attention_mask = _attn_row.narrow(
                0, 0, prefill_positions.shape[0]
            ).repeat(batch_size, 1)
            # that being said, this should always be a matrix of true during prefill,
            # so we can probably just use the mask directly.
        else:
            decoder_attention_mask = None

        # decoder_attention_mask = None

        finished, next_tokens, decoder_outputs = self.run_decoder(
            encoder_model_output,
            encoder_attention_mask,
            decoder_inputs,
            decoder_attention_mask,
            prefill_positions,
            past_decoder_outputs,
        )

        self.cache_position.add_(prefill_len)
        torch.narrow(self.next_tokens, 0, 0, batch_size).copy_(next_tokens)

        new_tokens = torch.cat((past_decoder_outputs, next_tokens), dim=1)
        return finished, new_tokens, decoder_outputs

    def decode(self, encoder_inputs, encoder_attention_mask, past_decoder_outputs):
        batch_size, encoder_sequence_length = encoder_inputs.shape

        if encoder_sequence_length > self.encoder_output.shape[1]:
            raise ValueError(
                "Encoder sequence length is greater than the max encoder sequence length"
            )
        if batch_size > self.encoder_output.shape[0]:
            raise ValueError("Batch size is greater than the max batch size")

        narrowed_encoder_output = torch.narrow(
            self.encoder_output, 0, 0, batch_size
        ).narrow(1, 0, encoder_sequence_length)
        encoder_model_output = BaseModelOutput(
            last_hidden_state=narrowed_encoder_output
        )

        decoder_inputs = torch.narrow(self.next_tokens, 0, 0, batch_size)
        finished, next_tokens, decoder_outputs = self.run_decoder(
            encoder_model_output,
            encoder_attention_mask,
            decoder_inputs,
            None,
            self.cache_position,
            past_decoder_outputs,
        )
        self.cache_position.add_(1)
        torch.narrow(self.next_tokens, 0, 0, batch_size).copy_(next_tokens)
        new_tokens = next_tokens

        return finished, new_tokens, decoder_outputs
