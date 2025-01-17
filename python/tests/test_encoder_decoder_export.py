# Load model directly

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from execu_tools.encoder_decoder_export import EncoderDecoderWrapper
from execu_tools.model_exporter import Exporter
from transformers.cache_utils import StaticCache, EncoderDecoderCache


def test_encoder_decoder_export(model_name="Helsinki-NLP/opus-mt-en-fr"):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name
        # attn_implementation=attn_implementation,
    )
    # torch._dynamo.config.capture_scalar_outputs = True

    cache_implementation = "static"
    max_batch_size = 1
    max_generation_length = 20

    # First get reference output using HuggingFace generate
    model.generation_config.update(
        use_cache=True,
        max_length=max_generation_length,
        num_beams=1,  # only greedy is compile compatible right now.
    )
    
    test_input = ["When the night has come and the land is dark, and the moon is the only light we will see."]
    input_ids = tokenizer(test_input, return_tensors="pt", padding=True)
    
    # Get reference output
    reference_output = model.generate(**input_ids)
    reference_text = tokenizer.decode(reference_output[0], skip_special_tokens=True)
    print(f"Reference output: {reference_text}")

    # Now test our wrapper implementation
    encoder_cache = StaticCache(
        model.config,
        max_cache_len=40,
        max_batch_size=1,
    )
    decoder_cache = StaticCache(
        model.config,
        max_cache_len=80,
        max_batch_size=1,
    )
    cache = EncoderDecoderCache(decoder_cache, encoder_cache)

    model_wrapper = EncoderDecoderWrapper(model, cache)

    finished = False
    finished, tokens = model_wrapper.generate(
        encoder_inputs=input_ids["input_ids"], reset_state=True
    )
    
    all_tokens = tokens
    while not finished:
        finished, new_tokens = model_wrapper.generate(
            encoder_inputs=input_ids["input_ids"], reset_state=False
        )
        all_tokens = torch.cat((all_tokens, new_tokens), dim=1)

    wrapper_text = tokenizer.decode(all_tokens[0], skip_special_tokens=True)
    print(f"Wrapper output: {wrapper_text}")
    
    # Assert the outputs match
    assert wrapper_text == reference_text, f"Expected: {reference_text}\nGot: {wrapper_text}"

def test_max_length_completion(model_name="Helsinki-NLP/opus-mt-en-fr"):
    """Test that generation stops when max_length is reached."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Set a very short max length to force truncation
    max_generation_length = 5
    model.generation_config.update(
        use_cache=True,
        max_length=max_generation_length,
        num_beams=1,
    )
    
    test_input = ["This is a long sentence that should generate a long output to test max length."]
    input_ids = tokenizer(test_input, return_tensors="pt", padding=True)
    
    # Setup cache and wrapper
    encoder_cache = StaticCache(model.config, max_cache_len=40, max_batch_size=1)
    decoder_cache = StaticCache(model.config, max_cache_len=80, max_batch_size=1)
    cache = EncoderDecoderCache(decoder_cache, encoder_cache)
    model_wrapper = EncoderDecoderWrapper(model, cache)
    
    # Generate tokens
    finished = False
    finished, tokens = model_wrapper.generate(encoder_inputs=input_ids["input_ids"], reset_state=True)
    all_tokens = tokens
    
    num_steps = 0
    while not finished:
        finished, new_tokens = model_wrapper.generate(
            encoder_inputs=input_ids["input_ids"], reset_state=False
        )
        all_tokens = torch.cat((all_tokens, new_tokens), dim=1)
        num_steps += 1
    
    # Verify we stopped at max length
    assert all_tokens.shape[1] <= max_generation_length, f"Generated {all_tokens.shape[1]} tokens, expected <= {max_generation_length}"

def test_eos_token_completion(model_name="Helsinki-NLP/opus-mt-en-fr"):
    """Test that generation stops when EOS token is generated."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Set a long max length to ensure we stop due to EOS
    max_generation_length = 50
    model.generation_config.update(
        use_cache=True,
        max_length=max_generation_length,
        num_beams=1,
    )
    
    test_input = ["Hello world"]  # Short input likely to generate EOS before max length
    input_ids = tokenizer(test_input, return_tensors="pt", padding=True)
    
    # Get reference output to verify EOS behavior
    reference_output = model.generate(**input_ids)
    reference_length = reference_output.shape[1]
    
    # Setup cache and wrapper
    encoder_cache = StaticCache(model.config, max_cache_len=40, max_batch_size=1)
    decoder_cache = StaticCache(model.config, max_cache_len=80, max_batch_size=1)
    cache = EncoderDecoderCache(decoder_cache, encoder_cache)
    model_wrapper = EncoderDecoderWrapper(model, cache)
    
    # Generate tokens
    finished = False
    finished, tokens = model_wrapper.generate(encoder_inputs=input_ids["input_ids"], reset_state=True)
    all_tokens = tokens
    
    while not finished:
        finished, new_tokens = model_wrapper.generate(
            encoder_inputs=input_ids["input_ids"], reset_state=False
        )
        all_tokens = torch.cat((all_tokens, new_tokens), dim=1)
    
    # Verify we stopped at same length as reference (due to EOS)
    assert all_tokens.shape[1] == reference_length, (
        f"Generated {all_tokens.shape[1]} tokens, expected {reference_length} "
        f"(reference output length)"
    )
    assert all_tokens.shape[1] < max_generation_length, (
        f"Generated {all_tokens.shape[1]} tokens, which equals max_length {max_generation_length}. "
        f"Expected to stop earlier due to EOS token."
    )

if __name__ == "__main__":
    test_encoder_decoder_export()
    test_max_length_completion()
    test_eos_token_completion()


# class M(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.param = torch.nn.Parameter(torch.rand(3, 4))
#         self.linear = torch.nn.Linear(4, 5)

#     def forward(self, x):
#         return self.linear(x + self.param).clamp(min=0.0, max=1.0)


# example_args = (torch.randn(3, 4),)
# pre_autograd_aten_dialect = export_for_training(M(), example_args).module()
# # Optionally do quantization:
# # pre_autograd_aten_dialect = convert_pt2e(prepare_pt2e(pre_autograd_aten_dialect, CustomBackendQuantizer))
# aten_dialect: ExportedProgram = export(pre_autograd_aten_dialect, example_args)
# edge_program: exir.EdgeProgramManager = exir.to_edge(aten_dialect)
# # Optionally do delegation:
# # edge_program = edge_program.to_backend(CustomBackendPartitioner)
# executorch_program: exir.ExecutorchProgramManager = edge_program.to_executorch(
#     ExecutorchBackendConfig(
#         passes=[],  # User-defined passes
#     )
# )

# with open("model.pte", "wb") as file:
#     file.write(executorch_program.buffer)

# def test_encoder_decoder(model_name="Helsinki-NLP/opus-mt-en-fr"):
#     """Validate that the model works at all.

#     Args:
#         model_name (str, optional): _description_. Defaults to "Helsinki-NLP/opus-mt-en-fr".
#     """
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSeq2SeqLM.from_pretrained(
#         model_name
#         # attn_implementation=attn_implementation,
#     )

#     cache_implementation = "static"
#     max_batch_size = 4
#     max_generation_length = 200

#     strings_1 = [
#         "When the night has come and the land is dark, and the moon is the only light we will see.",
#         "Abba is the best",
#         # "When the night has come and the land is dark, and the moon is the only light we will see.",
#         # "When the night has come and the land is dark, and the moon is the only light we will see.",
#     ]
#     input_ids = tokenizer(strings_1, return_tensors="pt", padding=True)
#     tokens = model.generate(**input_ids)
#     text_translated = [tokenizer.decode(t, skip_special_tokens=False) for t in tokens]
#     print(text_translated)
