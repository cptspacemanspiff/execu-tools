import pytest
from pathlib import Path
from transformers import AutoTokenizer
from execu_tools.tokenizer_converters import get_fast_tokenizer


@pytest.fixture
def english_tokenization_test_string():
    # grab testfile from current directory, using pathlib
    current_dir = Path(__file__).parent
    with open(current_dir / "sample_files" / "english_tokenization_test.txt", "r") as f:
        return f.read()


@pytest.mark.parametrize(
    "model_name",
    [
        # "facebook/mbart-large-50",
        # "facebook/mbart-large-cc25",
        "Helsinki-NLP/opus-mt-en-fr",
        # Add more model names as needed
    ],
)
def test_opus(model_name, english_tokenization_test_string):
    print(f"Testing model: {model_name}")
    print(english_tokenization_test_string)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    fast_tokenizer = get_fast_tokenizer(tokenizer)

    # Add your test assertions here
    assert fast_tokenizer is not None

    expected_tokens = tokenizer.encode(english_tokenization_test_string)
    # Example test: Check if tokenization works
    tokens = fast_tokenizer.encode(english_tokenization_test_string)
    assert expected_tokens == tokens.ids

    # Test batch encoding with padding
    batch_texts = [english_tokenization_test_string, "A shorter text"]
    batch_encoding = fast_tokenizer.encode_batch(
        batch_texts,
    )
    expected_batch = tokenizer(
        batch_texts, padding=True, truncation=True, return_tensors=None
    )

    assert len(batch_encoding) == 2
    for i in range(len(batch_encoding)):
        assert batch_encoding[i].ids == expected_batch["input_ids"][i]

    # Test batch decoding
    expected_batch_decoded = tokenizer.batch_decode(
        expected_batch["input_ids"],
        skip_special_tokens=False,
    )
    fast_batch_decoded = fast_tokenizer.decode_batch(
        [b.ids for b in batch_encoding], skip_special_tokens=False
    )
    
    # Strip any extra spaces after special tokens for comparison
    # these differ, but not functionally expected_batch_decoded has a space 
    # inserted after the <pad> token (and all other special tokens), while 
    # fast_batch_decoded does not.
    def clean_special_token_spaces(text):
        # Common special tokens that might have extra spaces
        # special_tokens = tokenizer.
        cleaned = text
        for token in tokenizer.special_tokens_map.values():
            cleaned = cleaned.replace(f"{token} ", token)
        return cleaned.strip()

    expected_cleaned = [clean_special_token_spaces(text) for text in expected_batch_decoded]
    fast_cleaned = [clean_special_token_spaces(text) for text in fast_batch_decoded]
    
    assert expected_cleaned == fast_cleaned

    # Test batch decoding with skip_special_tokens=True
    expected_batch_decoded_no_special = tokenizer.batch_decode(
        expected_batch["input_ids"],
        skip_special_tokens=True,
    )
    fast_batch_decoded_no_special = fast_tokenizer.decode_batch(
        [b.ids for b in batch_encoding], skip_special_tokens=True
    )
    
    assert expected_batch_decoded_no_special == fast_batch_decoded_no_special




if __name__ == "__main__":
    # List of models to test
    models = [
        # "facebook/mbart-large-50",
        # "facebook/mbart-large-cc25",
        "Helsinki-NLP/opus-mt-en-fr",
    ]

    # Get the test string directly (not using fixture)
    current_dir = Path(__file__).parent
    with open(current_dir / "sample_files" / "english_tokenization_test.txt", "r") as f:
        test_string = f.read()

    # Run test for each model
    for model in models:
        test_opus(model, test_string)
