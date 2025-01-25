
from transformers import AutoTokenizer
from execu_tools.tokenizer_export import export_MariamTokenizer


def test_opus(model_name="Helsinki-NLP/opus-mt-en-fr"):    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    export_MariamTokenizer(tokenizer)
    # expected_result = tokenizer.encode('Hello World')

    # actual result from spm directly:
    # tokens = tokenizer.encode('Hello World',out_type=int)


    pass


if __name__ == "__main__":
    test_opus()