# Load model directly

import copy
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from execu_tools.encoder_decoder_export import EncoderDecoderWrapper
from execu_tools.model_exporter import Exporter
from transformers.cache_utils import StaticCache, EncoderDecoderCache


def setup_model_and_tokenizer(
    model_name="Helsinki-NLP/opus-mt-en-fr", max_length=25
):  # this test fails if max_length is 20 (20 is a default value, use something else.)
    """Common setup for model, tokenizer and generation config."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if max_length == 20:
        raise ValueError(
            "max_length must not be 20, hf generate which is used for "
            "reference in testing has a special case for 20 (the default), "
            "and it behaves oddly."
        )

    model.generation_config.update(
        use_cache=True,
        max_length=max_length,
        num_beams=1,  # only greedy is compile compatible right now.
    )
    return model, tokenizer


def setup_wrapper(
    model, max_cache_len_encoder=40, max_cache_len_decoder=80, max_batch_size=1
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


def generate_with_wrapper(model_wrapper, input_ids, set_ones_after_reset=False):
    """Run generation with the wrapper until completion."""
    finished = False
    finished, tokens = model_wrapper.generate(
        encoder_inputs=input_ids["input_ids"],
        encoder_attention_mask=input_ids["attention_mask"],
        reset_state=True
    )
    all_tokens = tokens


    if set_ones_after_reset:
        model_wrapper.decoded_outputs[:,1:] = 1.0

    while not finished:
        finished, new_tokens = model_wrapper.generate(
            encoder_inputs=input_ids["input_ids"],
            encoder_attention_mask=input_ids["attention_mask"],
            reset_state=False
        )
        all_tokens = torch.cat((all_tokens, new_tokens), dim=1)

    return all_tokens


def test_encoder_decoder_export(model_name="Helsinki-NLP/opus-mt-en-fr"):
    max_generation_length = 25
    model, tokenizer = setup_model_and_tokenizer(model_name, max_generation_length)

    test_input = [
        "When the night has come and the land is dark, and the moon is the only light we will see."
    ]
    input_ids = tokenizer(test_input, return_tensors="pt", padding=True)

    # Get reference output
    reference_output = model.generate(**input_ids)
    reference_text = tokenizer.decode(reference_output[0], skip_special_tokens=False)
    print(f"Reference output: {reference_text}")

    # Test wrapper implementation
    model_wrapper = setup_wrapper(model)
    all_tokens = generate_with_wrapper(model_wrapper, input_ids)
    wrapper_text = tokenizer.decode(all_tokens[0], skip_special_tokens=False)
    print(f"Wrapper output: {wrapper_text}")

    assert (
        wrapper_text == reference_text
    ), f"Expected: {reference_text}\nGot: {wrapper_text}"


def test_max_length_completion(model_name="Helsinki-NLP/opus-mt-en-fr"):
    """Test that generation stops when max_length is reached."""
    max_generation_length = 5  # Very short to force truncation
    model, tokenizer = setup_model_and_tokenizer(model_name, max_generation_length)

    test_input = [
        "This is a long sentence that should generate a long output to test max length."
    ]
    input_ids = tokenizer(test_input, return_tensors="pt", padding=True)

    # Get reference output
    reference_output = model.generate(**input_ids)

    model_wrapper = setup_wrapper(model)
    all_tokens = generate_with_wrapper(model_wrapper, input_ids)

    # Check length constraint
    assert (
        all_tokens.shape[1] <= max_generation_length
    ), f"Generated {all_tokens.shape[1]} tokens, expected <= {max_generation_length}"

    # Check that the tokens match the reference up to max_length
    assert torch.equal(
        all_tokens[0, :max_generation_length],
        reference_output[0, :max_generation_length],
    ), (
        f"Generated tokens don't match reference up to max_length.\n"
        f"Generated: {all_tokens[0, :max_generation_length]}\n"
        f"Reference: {reference_output[0, :max_generation_length]}"
    )


def test_eos_token_completion(model_name="Helsinki-NLP/opus-mt-en-fr"):
    """Test that generation stops when EOS token is generated."""
    max_generation_length = 50  # Long enough to ensure EOS is reached
    model, tokenizer = setup_model_and_tokenizer(model_name, max_generation_length)

    test_input = ["Hello world"]  # Short input likely to generate EOS before max length
    input_ids = tokenizer(test_input, return_tensors="pt", padding=True)

    # Get reference output to verify EOS behavior
    reference_output = model.generate(**input_ids)

    model_wrapper = setup_wrapper(model)
    all_tokens = generate_with_wrapper(model_wrapper, input_ids)

    # Check that the sequence lengths match
    assert (
        all_tokens.shape[1] == reference_output.shape[1]
    ), f"Generated sequence length {all_tokens.shape[1]} doesn't match reference length {reference_output.shape[1]}"

    # Check that all tokens match exactly
    assert torch.equal(all_tokens[0], reference_output[0]), (
        f"Generated tokens don't match reference.\n"
        f"Generated: {all_tokens[0]}\n"
        f"Reference: {reference_output[0]}"
    )

    # Verify EOS token is present and at the same position
    eos_token_id = model.config.eos_token_id
    generated_eos_pos = (all_tokens[0] == eos_token_id).nonzero(as_tuple=True)[0]
    reference_eos_pos = (reference_output[0] == eos_token_id).nonzero(as_tuple=True)[0]

    assert torch.equal(generated_eos_pos, reference_eos_pos), (
        f"EOS token position mismatch.\n"
        f"Generated EOS at: {generated_eos_pos}\n"
        f"Reference EOS at: {reference_eos_pos}"
    )

    # Verify we stopped before max_length
    assert all_tokens.shape[1] < max_generation_length, (
        f"Generated {all_tokens.shape[1]} tokens, which equals max_length {max_generation_length}. "
        f"Expected to stop earlier due to EOS token."
    )


def test_batched_generation(model_name="Helsinki-NLP/opus-mt-en-fr"):
    """Test that generation works correctly with batched inputs."""
    max_generation_length = 25
    batch_size = 3
    model, tokenizer = setup_model_and_tokenizer(model_name, max_generation_length)

    test_inputs = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello world, how are you today?",
        "Python is a great programming language.",
    ]
    input_ids = tokenizer(test_inputs, return_tensors="pt", padding=True)

    # Get reference outputs
    reference_outputs = model.generate(**input_ids)
    reference_texts = [
        tokenizer.decode(output, skip_special_tokens=False)
        for output in reference_outputs
    ]
    print(f"Reference outputs: {reference_texts}")

    # Test wrapper implementation with batched inputs
    model_wrapper = setup_wrapper(model, max_batch_size=batch_size)
    all_tokens = generate_with_wrapper(model_wrapper, input_ids)
    wrapper_texts = [
        tokenizer.decode(tokens, skip_special_tokens=False)
        for tokens in all_tokens
    ]
    print(f"Wrapper outputs: {wrapper_texts}")

    # Check that each sequence in the batch matches the reference
    for i, (wrapper_text, reference_text) in enumerate(zip(wrapper_texts, reference_texts)):
        assert wrapper_text == reference_text, (
            f"Batch item {i} mismatch:\n"
            f"Expected: {reference_text}\n"
            f"Got: {wrapper_text}"
        )


def test_ones_cache_generation(model_name="Helsinki-NLP/opus-mt-en-fr"):
    """Test that generation works correctly when the static cache is initialized with ones."""
    max_generation_length = 40
    batch_size = 2
    model, tokenizer = setup_model_and_tokenizer(model_name, max_generation_length)

    # Use shared setup and then modify cache
    model_wrapper = setup_wrapper(model, max_batch_size=batch_size)
    
    # Set all cache buffers to ones
    for cache in [model_wrapper.cache.self_attention_cache, model_wrapper.cache.cross_attention_cache]:
        for key_cache, value_cache in zip(cache.key_cache, cache.value_cache):
            key_cache.fill_(1.0)
            value_cache.fill_(1.0)

    model_wrapper.decoded_outputs.fill_(1.0)

    # Test with multiple inputs to ensure batch processing works with ones cache
    test_inputs = [
        "This is a ", #short fragment for validating attention mask.
        "Another test sentence for verification of ones in the cache."
    ]
    input_ids = tokenizer(test_inputs, return_tensors="pt", padding=True)

    # Get reference outputs
    reference_outputs = model.generate(**input_ids)
    reference_texts = [
        tokenizer.decode(output, skip_special_tokens=False)
        for output in reference_outputs
    ]
    print(f"Reference outputs with normal cache: {reference_texts}")

    # Test wrapper implementation with ones cache
    all_tokens = generate_with_wrapper(model_wrapper, input_ids, set_ones_after_reset=False)
    wrapper_texts = [
        tokenizer.decode(tokens, skip_special_tokens=False)
        for tokens in all_tokens
    ]
    print(f"Wrapper outputs with ones cache: {wrapper_texts}")

    # Check that each sequence in the batch matches the reference
    for i, (wrapper_text, reference_text) in enumerate(zip(wrapper_texts, reference_texts)):
        assert wrapper_text == reference_text, (
            f"Batch item {i} mismatch with ones cache:\n"
            f"Expected: {reference_text}\n"
            f"Got: {wrapper_text}"
        )


if __name__ == "__main__":
    # test_encoder_decoder_export()
    # test_max_length_completion()
    # test_eos_token_completion()
    # test_batched_generation()
    test_ones_cache_generation()


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
