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


def generate_with_wrapper(model_wrapper : EncoderDecoderWrapper, input_ids, set_ones_after_reset=False, compile=False):
    """Run generation with the wrapper until completion."""
    with torch.no_grad():
        if compile:
            model_wrapper.reset_encode_prefill = torch.compile(model_wrapper.reset_encode_prefill, mode="reduce-overhead",fullgraph=True)
            model_wrapper.decode = torch.compile(model_wrapper.decode, mode="reduce-overhead",fullgraph=True)

        finished = False
        finished, tokens, decoder_outputs = model_wrapper.reset_encode_prefill(
            encoder_inputs=input_ids["input_ids"],
            encoder_attention_mask=input_ids["attention_mask"],
            prefill_prompt=model_wrapper.format_prompt()
        )
        all_tokens = tokens

        while not finished:
            finished, new_tokens, decoder_outputs = model_wrapper.decode(
                encoder_inputs=input_ids["input_ids"],
                encoder_attention_mask=input_ids["attention_mask"],
                past_decoder_outputs=decoder_outputs
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


def test_sentence_fragment_cache_generation(model_name="Helsinki-NLP/opus-mt-en-fr"):
    """Test that generation works correctly when the static cache is initialized with ones."""
    max_generation_length = 40
    batch_size = 4 #differs from input batch size to test batching.
    model, tokenizer = setup_model_and_tokenizer(model_name, max_generation_length)

    # Use shared setup and then modify cache
    model_wrapper = setup_wrapper(model, max_batch_size=batch_size)

    # Test with multiple inputs to ensure batch processing.
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
    print(f"Reference outputs: {reference_texts}")

    # Test wrapper implementation with ones cache
    all_tokens = generate_with_wrapper(model_wrapper, input_ids, set_ones_after_reset=False)
    wrapper_texts = [
        tokenizer.decode(tokens, skip_special_tokens=False)
        for tokens in all_tokens
    ]
    print(f"Wrapper outputs: {wrapper_texts}")

    # Check that each sequence in the batch matches the reference
    for i, (wrapper_text, reference_text) in enumerate(zip(wrapper_texts, reference_texts)):
        assert wrapper_text == reference_text, (
            f"Batch item {i} mismatch with ones cache:\n"
            f"Expected: {reference_text}\n"
            f"Got: {wrapper_text}"
        )


def test_compiled_generation(model_name="Helsinki-NLP/opus-mt-en-fr"):
    """Test that generation works correctly with torch.compile."""
    max_generation_length = 50
    model, tokenizer = setup_model_and_tokenizer(model_name, max_generation_length)

    test_input = [
        "When the night has come and the land is dark, and the moon is the only light we will see."
    ]
    input_ids = tokenizer(test_input, return_tensors="pt", padding=True)

    # Get reference output from uncompiled model
    model_wrapper = setup_wrapper(model)
    reference_tokens = generate_with_wrapper(model_wrapper, input_ids, compile=False)
    reference_text = tokenizer.decode(reference_tokens[0], skip_special_tokens=False)
    print(f"Reference output (uncompiled): {reference_text}")

    # Test with compiled generate function
    compiled_tokens = generate_with_wrapper(model_wrapper, input_ids, compile=True)
    compiled_text = tokenizer.decode(compiled_tokens[0], skip_special_tokens=False)
    print(f"Wrapper output (compiled): {compiled_text}")

    assert (
        compiled_text == reference_text
    ), f"Expected: {reference_text}\nGot: {compiled_text}"


def test_compiled_generation_batch_mismatch(model_name="Helsinki-NLP/opus-mt-en-fr"):
    """Test that compiled generation works correctly with batch size smaller than cache max batch size."""
    max_generation_length = 25
    cache_batch_size = 4  # Larger max batch size for cache
    input_batch_size = 2  # Smaller actual input batch size
    model, tokenizer = setup_model_and_tokenizer(model_name, max_generation_length)

    test_inputs = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a great programming language.",
    ]
    input_ids = tokenizer(test_inputs, return_tensors="pt", padding=True)

    # Create wrapper with larger max batch size
    model_wrapper = setup_wrapper(model, max_batch_size=cache_batch_size)

    # Get reference output from uncompiled model
    reference_tokens = generate_with_wrapper(model_wrapper, input_ids, compile=False)
    reference_texts = [
        tokenizer.decode(tokens, skip_special_tokens=False)
        for tokens in reference_tokens
    ]
    print(f"Reference outputs (uncompiled): {reference_texts}")

    # Test with compiled generate function
    compiled_tokens = generate_with_wrapper(model_wrapper, input_ids, compile=True)
    compiled_texts = [
        tokenizer.decode(tokens, skip_special_tokens=False)
        for tokens in compiled_tokens
    ]
    print(f"Wrapper outputs (compiled): {compiled_texts}")

    # Check that each sequence in the batch matches the reference
    for i, (compiled_text, reference_text) in enumerate(zip(compiled_texts, reference_texts)):
        assert compiled_text == reference_text, (
            f"Batch item {i} mismatch:\n"
            f"Expected: {reference_text}\n"
            f"Got: {compiled_text}"
        )

    # Verify batch sizes
    assert input_ids["input_ids"].shape[0] == input_batch_size, "Input batch size mismatch"
    assert compiled_tokens.shape[0] == input_batch_size, "Output batch size mismatch"


if __name__ == "__main__":
    # test_encoder_decoder_export()
    # test_max_length_completion()
    # test_eos_token_completion()
    # test_batched_generation()
    test_sentence_fragment_cache_generation()
    # test_compiled_generation()
    # test_compiled_generation_batch_mismatch()
    pass