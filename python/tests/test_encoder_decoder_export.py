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


def generate_with_wrapper(model_wrapper, input_ids, set_ones_after_reset=False, compile=False):
    """Run generation with the wrapper until completion."""
    generate_fn = model_wrapper.generate
    if compile:
        with torch.no_grad():
            compiled_generate_fn = torch.compile(generate_fn, mode="reduce-overhead",fullgraph=True)
            model_wrapper.generate = compiled_generate_fn
            generate_fn = model_wrapper.generate

    finished = False
    finished, tokens, decoder_outputs = generate_fn(
        encoder_inputs=input_ids["input_ids"],
        encoder_attention_mask=input_ids["attention_mask"],
        reset_state=True,
        past_decoder_outputs=torch.tensor([[]])
    )
    all_tokens = tokens

    if set_ones_after_reset:
        model_wrapper.decoded_outputs[:,1:] = 1.0

    while not finished:
        finished, new_tokens, decoder_outputs = generate_fn(
            encoder_inputs=input_ids["input_ids"],
            encoder_attention_mask=input_ids["attention_mask"],
            reset_state=False,
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


def manual_test_compiled_generation(model_name="Helsinki-NLP/opus-mt-en-fr"):
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

    # assert (
    #     compiled_text == reference_text
    # ), f"Expected: {reference_text}\nGot: {compiled_text}"

    # # Test with a different input to ensure compilation works consistently
    # test_input_2 = ["This is another test sentence to verify compilation works consistently."]
    # input_ids_2 = tokenizer(test_input_2, return_tensors="pt", padding=True)
    
    # reference_tokens_2 = generate_with_wrapper(model_wrapper, input_ids_2, compile=False)
    # reference_text_2 = tokenizer.decode(reference_tokens_2[0], skip_special_tokens=False)
    
    # compiled_tokens_2 = generate_with_wrapper(model_wrapper, input_ids_2, compile=True)
    # compiled_text_2 = tokenizer.decode(compiled_tokens_2[0], skip_special_tokens=False)
    
    # assert (
    #     compiled_text_2 == reference_text_2
    # ), f"Expected (2nd input): {reference_text_2}\nGot: {compiled_text_2}"


if __name__ == "__main__":
    test_encoder_decoder_export()
    # test_max_length_completion()
    # test_eos_token_completion()
    # test_batched_generation()
    # test_ones_cache_generation()
    # test_compiled_generation()
    pass