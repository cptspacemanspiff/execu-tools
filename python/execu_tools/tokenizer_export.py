# Given a tokenizer, wraps up either sentencepiece or huggingface blobs.



from enum import Enum

from transformers.convert_slow_tokenizer import SpmConverter
class TokenizerType(Enum):
    HuggingFace = 0
    SentencePiece = 1

class TokenizerBlob:
    def to_bytes(self) -> bytes:
        raise NotImplementedError()

    def from_bytes(self, blob: bytes):
        raise NotImplementedError()

class EncoderDecoderTokenizerWrapper:
    Type: TokenizerType
    shared_tokenizer : bool
    encoder_tokenizer : TokenizerBlob
    decoder_tokenizer : TokenizerBlob




class MarianConverter(SpmConverter):
    def __init__(self,tokenizer_spm_file):
        self.original_tokenizer = tokenizer_spm_file
        super().__init__(tokenizer_spm_file)



def export_MariamTokenizer(tokenizer) -> EncoderDecoderTokenizerWrapper:
    from transformers.models.marian.tokenization_marian import MarianTokenizer
    import sentencepiece as spm


    
    tokenizer : MarianTokenizer = tokenizer

    sp = spm.SentencePieceProcessor()
    sp.load('/home/nlong/execu-tools/opus/source.spm')



    sp.Encode('Hello World',add_eos=True)

    encoded = tokenizer.encode('Hello World',add_eos=True)

    # tokenizer.encode_tokens_to_ids(['Hello World'])

    result = tokenizer.decode(tokenizer.encode('Hello World',add_eos=True))

    tokenizer.vocab_file = tokenizer.spm_files[0]
    en_encoder_tokenizer = MarianConverter(tokenizer).converted()
    tokenizer.vocab_file = tokenizer.spm_files[1]
    fr_decoder_tokenizer = MarianConverter(tokenizer).converted()

    shared_vocab_dict = tokenizer.get_vocab()

    
    en_dict = en_encoder_tokenizer.get_vocab()
    en_map = {}
    num_unk = 0
    for key, value in en_dict.items():
        if key in shared_vocab_dict:
            en_map[value] = shared_vocab_dict[key]
        else:
            en_map[value] = shared_vocab_dict['<unk>']
            num_unk += 1
            print(f"Key {key} not found in shared_vocab_dict")
    print(f"Number of unk tokens in en: {num_unk}")

    fr_dict = fr_decoder_tokenizer.get_vocab()
    fr_map = {}
    num_unk = 0
    for key, value in fr_dict.items():
        if key in shared_vocab_dict:
            fr_map[value] = shared_vocab_dict[key]
        else:
            fr_map[value] = shared_vocab_dict['<unk>']
            num_unk += 1
            print(f"Key {key} not found in shared_vocab_dict")
    print(f"Number of unk tokens in fr: {num_unk}")


    en_encoder_tokenizer.encode_tokens_to_ids(['Hello World'])


    # MarianTokenizer is a sentencepiece tokenizer w/ 2 spm files
    # source_spm_file = source.spm
    # target_spm_file = target.spm



    # encode_spm_file = 

    # get the tokenizer type:
    if not isinstance(tokenizer, MarianTokenizer):
        tokenizer_type = TokenizerType.SentencePiece


    pass








