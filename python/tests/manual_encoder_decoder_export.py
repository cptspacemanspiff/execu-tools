import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.cache_utils import StaticCache, EncoderDecoderCache
from execu_tools.encoder_decoder_export import EncoderDecoderWrapper
from execu_tools.model_exporter import Exporter
from torch.export import Dim
from pathlib import Path

def setup_model_and_tokenizer(model_name="Helsinki-NLP/opus-mt-en-fr", max_length=25):
    """Setup model and tokenizer with specified parameters."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    model.generation_config.update(
        use_cache=True,
        max_length=max_length,
        num_beams=1,  # only greedy search is supported for now
    )
    return model, tokenizer

def setup_wrapper(model, max_cache_len_encoder=40, max_cache_len_decoder=80, max_batch_size=1):
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
    # setup vars:
    max_cache_len_encoder = 40
    max_cache_len_decoder = 40
    max_batch_size = 4

    # Setup model and wrapper
    model, tokenizer = setup_model_and_tokenizer()
    model_wrapper = setup_wrapper(model, max_cache_len_encoder=max_cache_len_encoder, 
                                max_cache_len_decoder=max_cache_len_decoder, 
                                max_batch_size=max_batch_size)
    
    # Prepare example inputs
    test_input = ["Hello world", "Hello world 2", "Hello world 3"]
    input_ids = tokenizer(test_input, return_tensors="pt", padding=True)
    
    # Initialize exporter
    exporter = Exporter(model_wrapper)
    
    # Define dynamic dimensions
    batch_dim = Dim("batch_size", min=1, max=max_batch_size)
    encoder_seq_len_dim = Dim("encoder_seq_length", min=1, max=max_cache_len_encoder)
    decoder_seq_len_dim = Dim("decoder_seq_length", min=0, max=max_cache_len_decoder)
    
    # Create example inputs for tracing with dynamic dimensions
    example_inputs = {
        "encoder_inputs": (input_ids["input_ids"], {0: batch_dim, 1: encoder_seq_len_dim}),
        "encoder_attention_mask": (input_ids["attention_mask"], {0: batch_dim, 1: encoder_seq_len_dim}),
        "past_decoder_outputs": (torch.zeros(input_ids["input_ids"].shape[0], 2), {0: batch_dim, 1: decoder_seq_len_dim}) # must have same batch size as encoder inputs
    }

    # Register the forward method with dynamic dimensions
    exporter.register(model_wrapper.forward, **example_inputs)

    # Export the model through different stages
    exported_model = exporter.export()

    test_input_1 = {'encoder_inputs': torch.ones(3, 3, dtype=torch.long),
                   'encoder_attention_mask': torch.ones(3, 3, dtype=torch.long),
                   'past_decoder_outputs': torch.zeros(3, 2, dtype=torch.long)}
    
    test_input_2 = {'encoder_inputs': torch.ones(3, 2, dtype=torch.long),
                   'encoder_attention_mask': torch.ones(3, 2, dtype=torch.long),
                   'past_decoder_outputs': torch.zeros(3, 0, dtype=torch.long)}
    
    # validate that we can run it with the dynamic dimensions:
    # validate that we can run it with the dynamic dimensions:
    model_wrapper.forward(**example_inputs)

    # exporter.to_edge()
    # exporter.to_executorch()

    # # Save the exported program
    # output_dir = Path(__file__).parent / "export_artifacts"
    # exporter.save(output_dir, "encoder_decoder_model")
    
    # print("Model exported successfully to encoder_decoder_model.pte")

if __name__ == "__main__":
    export_model()
