from contextlib import contextmanager
import torch

# from transfomers
from dataclasses import dataclass
from transformers import PreTrainedModel

from transformers import StaticCache, EncoderDecoderCache


@contextmanager
def patch_forward(obj: torch.nn.Module, new_method):
    """
    Patches the forward method of a PyTorch module, needs to patch the class not
    just the instance b/c otherwise export will error out when trying to produce
    items that reference self.
    """
    original_method = obj.__class__.forward
    obj.__class__.forward = new_method
    try:
        yield
    finally:
        # Restore the original method
        obj.__class__.forward = original_method


@dataclass
class EncoderDecoderExportableConfig:
    ### Setup sizes:
    min_batch_size: int
    max_batch_size: int

    min_encoder_seq_len: int
    max_encoder_seq_len: int

    min_decoder_seq_len: int
    max_decoder_seq_len: int

    ### Cache setup:
    cache_dtype: torch.dtype


# def register_moth


class BaseExportableModel(torch.nn.Module):
    def __init__(self, torch_model: torch.nn.Module):
        super().__init__()

        self._registered_fn = {}

    def register_function(self, func):
        #
        self._registered_fn = func

    # def add_compile function:


class StatefulModel(torch.nn.Module):
    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.register_buffer(
            "cache", torch.zeros((max_batch_size, max_seq_len), dtype=torch.float32)
        )

    def set_cache(self, data: torch.Tensor):
        # get the shape of the date:
        data_shape = data.shape
        # Dynamically slice based on the target shape
        slices = tuple(slice(0, dim) for dim in data_shape)
        self.cache[slices] = data

    def get_cache(self, data: torch.Tensor):
        # load data this does not work because the data object we are assigning to is assumed to be static sized, not dynamic.
        # data[:data.shape[0], :data.shape[1]] = self.cache[: data.shape[0], : data.shape[1]]
        # batch_size = batch_size
        # seq_len = data.shape[1]
        shape = data.shape
        batch_sliced = self.cache.narrow(0,0, shape[0])
        seq_sliced = batch_sliced.narrow(1,0, shape[1])
        data[:shape[0], :shape[1]] = seq_sliced
        # return seq_sliced


class EncoderDecoderExportable(torch.nn.Module):

    def __init__(
        self,
        model: PreTrainedModel,
        exportable_config: EncoderDecoderExportableConfig,
    ):

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
            dtype=self.config.cache_dtype,
        )
        decoder_cache = StaticCache(
            model.config,
            max_cache_len=self.config.max_batch_size,
            max_batch_size=self.config.max_encoder_seq_len,
            dtype=self.config.cache_dtype,
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
        encoder_input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ):

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

        return encoder_output.last_hidden_state

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
        pass
