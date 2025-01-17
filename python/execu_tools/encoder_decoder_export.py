import torch
from transformers.modeling_outputs import BaseModelOutput
from transformers.cache_utils import EncoderDecoderCache
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList


class EncoderDecoderWrapper(torch.nn.Module):
    def __init__(self, model, cache):
        super().__init__()
        self.model = model
        self.cache = cache
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

        self.register_buffer("cache_position", torch.zeros((1,), dtype=torch.long))

        self.register_buffer(
            "next_tokens", torch.zeros((max_batch_size, 1), dtype=torch.long)
        )  # should be 2d, with batch size as first dim.

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

    def _process_next_tokens(self, batch_size, cache_position, next_token_scores, prev_decoder_outputs):
        # greedy search
        next_tokens = torch.argmax(next_token_scores, dim=-1)
        
        # Create new decoder outputs by concatenating previous outputs with next tokens
        decoder_outputs = torch.cat([prev_decoder_outputs, next_tokens.unsqueeze(1)], dim=1)
        
        # handle stopping criteria
        if self.has_eos_stopping_criteria:
            next_tokens = (
                next_tokens * self.unfinished_sequences
                + self.generation_config._pad_token_tensor
                * (1 - self.unfinished_sequences)
            )[:batch_size]

        self.unfinished_sequences = (
            self.unfinished_sequences
            & ~self.prepared_stopping_criteria(
                decoder_outputs, next_token_scores
            )
        )
        finished = self.unfinished_sequences.max() == 0
        
        return finished, next_tokens.unsqueeze(1), decoder_outputs

    def run_decoder(
        self, encoder_outputs, encoder_attention_mask, decoder_inputs, cache_position: torch.Tensor, prev_decoder_outputs
    ) -> dict:
        batch_size, seqlen = decoder_inputs.shape
        attn_mask = (
            self.decoder_attention_mask[cache_position, :seqlen]
            if self.is_causal
            else None
        )

        decoder_output = self.model(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_inputs,
            decoder_attention_mask=attn_mask,
            cache_position=cache_position,
            past_key_values=self.cache,
            return_dict=True,
        )

        # get the next token logits and process them
        next_token_logits = decoder_output.logits[:, -1, :].clone().float()
        next_token_scores = self.prepared_logits_processor(
            prev_decoder_outputs, next_token_logits
        )

        finished, next_tokens, decoder_outputs = self._process_next_tokens(
            batch_size, cache_position, next_token_scores, prev_decoder_outputs
        )
        
        return finished, next_tokens, decoder_outputs

    def generate(
        self, encoder_inputs, encoder_attention_mask, reset_state: bool = True, past_decoder_outputs=None
    ):

        # call encoder forward:
        # if this is the first call:
        batch_size, encoder_sequence_length = encoder_inputs.shape

        if encoder_sequence_length > self.encoder_output.shape[1]:
            raise ValueError(
                "Encoder sequence length is greater than the max encoder sequence length"
            )
        if batch_size > self.encoder_output.shape[0]:
            raise ValueError("Batch size is greater than the max batch size")

        if reset_state:
            self.cache.reset()

            # reset the unfinished sequences, up to our current batch size:
            self.unfinished_sequences.fill_(0)
            self.unfinished_sequences[:batch_size] = torch.ones(
                batch_size, dtype=torch.long
            )

            encoder = self.model.get_encoder()
            encoder_dict = encoder(
                input_ids=encoder_inputs,
                attention_mask=encoder_attention_mask,
            )
            self.encoder_output[:batch_size, :encoder_sequence_length, :] = (
                encoder_dict["last_hidden_state"]
            )
            encoder_model_output = BaseModelOutput(
                last_hidden_state=self.encoder_output[
                    :batch_size, :encoder_sequence_length, :
                ]
            )

            # We are doing prefill:
            prefill_len = 1
            prefill_positions = torch.arange(prefill_len)
            decoder_inputs = torch.tensor(
                [[self.generation_config._decoder_start_token_tensor]] * batch_size
            )

            # Initialize past_decoder_outputs for the first token
            if past_decoder_outputs.shape[1] == 0:
                past_decoder_outputs = decoder_inputs

            finished, next_tokens, decoder_outputs = self.run_decoder(
                encoder_model_output, encoder_attention_mask, decoder_inputs, 
                prefill_positions, past_decoder_outputs
            )

            self.cache_position = prefill_positions[-1:] + 1
            self.next_tokens = next_tokens

            new_tokens = torch.cat((past_decoder_outputs, next_tokens), dim=1)
            return finished, new_tokens, decoder_outputs

        else:
            # we are doing decode:
            encoder_model_output = BaseModelOutput(
                last_hidden_state=self.encoder_output[
                    :batch_size, :encoder_sequence_length, :
                ]
            )

            decoder_inputs = self.next_tokens
            finished, next_tokens, decoder_outputs = self.run_decoder(
                encoder_model_output, encoder_attention_mask, decoder_inputs, 
                self.cache_position, past_decoder_outputs
            )
            self.cache_position += 1
            self.next_tokens = next_tokens
            new_tokens = next_tokens

            return finished, new_tokens, decoder_outputs
