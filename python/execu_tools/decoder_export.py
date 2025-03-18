import torch
from transformers.cache_utils import StaticCache


class DecoderWrapper(torch.nn.Module):
    def __init__(self, model, cache):
        super().__init__()
        self.model = model
        self.cache = cache

        self.shared_buffer_fqns = []
        # add the cache:
        self.cache = cache
        self.shared_buffer_fqns.append("cache")

        if type(self.cache) is StaticCache:
            self.max_decoder_sequence_length = self.cache.key_cache[0].shape[2]
            self.max_batch_size = self.cache.key_cache[0].shape[0]
        else:
            raise ValueError("Cache is not a StaticCache")

        self.register_buffer("cache_position", torch.zeros((1,), dtype=torch.long))
        self.shared_buffer_fqns.append("cache_position")

        self.model._prepare_generation_config(
            self.model.generation_config, past_key_values=self.cache
        )

        

    def format_prompt(self, prompt=None):
        pass

    def prefill(self, prefill_tokens):
        pass

    def decoder(self, past_decoder_outputs):
        pass
