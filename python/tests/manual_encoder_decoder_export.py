import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.cache_utils import StaticCache, EncoderDecoderCache
from execu_tools.encoder_decoder_export import EncoderDecoderWrapper
from execu_tools.model_exporter import MultiEntryPointExporter, MethodArg
from execu_tools.tokenizer_converters import get_fast_tokenizer
from torch.export import Dim
from pathlib import Path
import copy


def setup_model_and_tokenizer(model_name="Helsinki-NLP/opus-mt-en-fr", max_length=25):
    """Setup model and tokenizer with specified parameters."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    model.generation_config.update(
        use_cache=True,
        max_length=max_length,
        num_beams=1,  # only greedy search is supported for now
    )
    return model, tokenizer


def setup_wrapper(
    model,
    tokenizer,
    max_cache_len_encoder=40,
    max_cache_len_decoder=80,
    max_batch_size=1,
):
    """Create cache and wrapper with given parameters."""
    encoder_cache = StaticCache(
        model.config,
        max_cache_len=max_cache_len_encoder,
        max_batch_size=max_batch_size,
    )
    decoder_cache = StaticCache(
        model.config,
        max_cache_len=max_cache_len_decoder,
        max_batch_size=max_batch_size,
    )
    cache = EncoderDecoderCache(decoder_cache, encoder_cache)
    return EncoderDecoderWrapper(model, cache)


def export_model():
    with torch.no_grad():
        # setup vars:
        max_cache_len_encoder = 20
        max_cache_len_decoder = 40
        max_batch_size = 4

        # Setup model and wrapper
        model, tokenizer = setup_model_and_tokenizer()
        model_wrapper = setup_wrapper(
            model,
            tokenizer,
            max_cache_len_encoder=max_cache_len_encoder
            + 1,  # TODO: why do we need to add 1 here?
            max_cache_len_decoder=max_cache_len_decoder,
            max_batch_size=max_batch_size,
        )

        constant_dict = {}

        fast_tokenizer = get_fast_tokenizer(tokenizer)
        fast_tokenizer_bytes = bytes(fast_tokenizer.to_str(), "utf-8")
        constant_dict["tokenizer_blob"] = torch.frombuffer(
            copy.copy(fast_tokenizer_bytes), dtype=torch.uint8
        )

        # Initialize exporter
        exporter = MultiEntryPointExporter(model_wrapper)

        # Define dynamic dimensions
        batch_dim = Dim("batch_size", min=1, max=max_batch_size)
        encoder_seq_len_dim = Dim(
            "encoder_seq_length", min=1, max=max_cache_len_encoder
        )  # TODO for some reason we cannot use the whole cross attention cache?
        decoder_seq_len_dim = Dim(
            "decoder_seq_length", min=1, max=max_cache_len_decoder
        )

        # Create example inputs for tracing with dynamic dimensions
        # TODO: max batch size does not work.
        example_batch_size = max_batch_size - 1 if max_batch_size > 1 else 1
        example_encoder_seq_len = (
            max_cache_len_encoder - 1 if max_cache_len_encoder > 1 else 1
        )
        example_decoder_seq_len = (
            max_cache_len_decoder - 1 if max_cache_len_decoder > 1 else 1
        )

        export_example_reset_encode_prefill = {
            "encoder_inputs": MethodArg(
                torch.ones(
                    example_batch_size, example_encoder_seq_len, dtype=torch.int
                ),
                {0: batch_dim, 1: encoder_seq_len_dim},
            ),
            "encoder_attention_mask": MethodArg(
                torch.ones(
                    example_batch_size, example_encoder_seq_len, dtype=torch.int
                ),
                {0: batch_dim, 1: encoder_seq_len_dim},
            ),
            "prefill_prompt": MethodArg(model_wrapper.format_prompt(), {}),
        }

        export_example_decode = {
            "encoder_inputs": MethodArg(
                torch.ones(
                    example_batch_size, example_encoder_seq_len, dtype=torch.int
                ),
                {0: batch_dim, 1: encoder_seq_len_dim},
            ),
            "encoder_attention_mask": MethodArg(
                torch.ones(
                    example_batch_size, example_encoder_seq_len, dtype=torch.bool
                ),
                {0: batch_dim, 1: encoder_seq_len_dim},
            ),
            "past_decoder_outputs": MethodArg(
                torch.zeros(
                    example_batch_size, example_decoder_seq_len, dtype=torch.int
                ),
                {0: batch_dim, 1: decoder_seq_len_dim},
            ),
        }

        input_example_reset_encode_prefill = {
            key: value.example_input
            for key, value in export_example_reset_encode_prefill.items()
        }
        model_wrapper.reset_encode_prefill(**input_example_reset_encode_prefill)

        input_example_decode = {
            key: value.example_input for key, value in export_example_decode.items()
        }
        model_wrapper.decode(**input_example_decode)

        # get a list of fqn to register as shared from the model_wrapper.
        exporter.register_shared_buffers(model_wrapper.get_shared_fqn())

        # Register the methods
        exporter.register(
            model_wrapper.reset_encode_prefill, **export_example_reset_encode_prefill
        )
        exporter.register(model_wrapper.decode, **export_example_decode)

        # Export the model through different stages
        exporter.export()
        print(
            f"Successfully exported model functions: {exporter.registered_method_dict.keys()}"
        )
        exporter.to_edge(constant_methods=constant_dict)
        print(f"Successfully converted to edge")
        exporter.to_executorch()
        print(f"Successfully converted to executorch")
        # # Save the exported program
        output_dir = Path(__file__).parent / "export_artifacts"
        exporter.save(output_dir, "opus_encoder_decoder_model")


if __name__ == "__main__":
    export_model()
