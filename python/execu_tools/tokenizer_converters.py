# Given a tokenizer, wraps up either sentencepiece or huggingface blobs.


from enum import Enum
from typing import Literal

from tokenizers import AddedToken, Tokenizer
from tokenizers.models import Unigram
from tokenizers.processors import TemplateProcessing

from transformers.convert_slow_tokenizer import SpmConverter
from transformers.models.marian.tokenization_marian import MarianTokenizer
from transformers import AutoTokenizer


class MarianConverter(SpmConverter):

    def __init__(
        self,
        tokenizer: MarianTokenizer,
        type: Literal["encoder", "decoder"] = "encoder",
    ):

        # MarianTokenizer is a sentencepiece unigram tokenizer with 2 sentencepiece files
        # source_spm_file = source.spm -> unigram encoder for source language
        # target_spm_file = target.spm -> unigram encoder for target language
        # After encoding w/unigram the tokens are remapped to the shared vocab
        # and fed to the model.

        # for decode, we do not use the unigram encoder, we use a wordlevel
        # model w/ the shared vocab.

        if type == "encoder":
            self.shared_vocab_dict = tokenizer.get_vocab()
            tokenizer.vocab_file = tokenizer.spm_files[0]
        elif type == "decoder":
            tokenizer.vocab_file = tokenizer.spm_files[1]  # target lang spm file.
        else:
            raise ValueError(
                f"Invalid type: {type} must be one of ['encoder', 'decoder']"
            )

        super().__init__(tokenizer)

    def vocab(self, proto):
        # reorder the vocab to match the shared vocab (which is larger)
        vocab = []
        # first fill the vocab with the shared vocab, use -1000 for all tokens that are not used.

        # get the spm vocab
        spm_vocab_list = [(piece.piece, piece.score) for piece in proto.pieces[:]]
        # convert to dict
        spm_vocab_dict = {piece: score for piece, score in spm_vocab_list}

        shared_vocab_dict: dict[str, int] = self.original_tokenizer.get_vocab()
        # make sure the shared vocab is sorted by the token id
        shared_vocab_dict = dict(sorted(shared_vocab_dict.items(), key=lambda x: x[1]))
        for token in shared_vocab_dict.items():
            if token[0] in spm_vocab_dict:
                vocab.append((token[0], spm_vocab_dict[token[0]]))
            else:
                vocab.append((token[0], -1000))

        return vocab

    def tokenizer(self, proto):
        model_type = proto.trainer_spec.model_type
        vocab_scores = self.vocab(proto)

        if model_type == 1:
            tokenizer = Tokenizer(
                Unigram(
                    vocab_scores,
                    unk_id=self.unk_id(proto),
                    byte_fallback=self.handle_byte_fallback,
                )
            )

        elif model_type == 2:
            raise NotImplementedError(
                "MarianConverter has not been tested with BPE sentence piece tokenizers."
            )

        else:
            raise Exception(
                "You're trying to run a `Unigram` model but you're file was trained with a different algorithm"
            )

        # ignore the special tokens in the sentencepiece model, they are either
        # already in the vocab or unknown.
        special_tokens = [
            token for token in self.original_tokenizer.special_tokens_map.values()
        ]
        tokenizer.add_tokens(
            [
                AddedToken(token, normalized=False, special=True)
                for token in special_tokens
            ]
        )

        return tokenizer

    def unk_id(self, proto):
        unk_token = self.original_tokenizer.unk_token
        unk_id = self.original_tokenizer.convert_tokens_to_ids(unk_token)
        return unk_id

    def post_processor(self):
        TemplateTokens = TemplateProcessing(
            single="$A </s>",
            pair="$A $B </s>",  # We do not handle pairs, so we just pad the second sentence.
            special_tokens=[
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )
        return TemplateTokens


# Note: MarianMT (Or at least OPUSMT) models have 2 tokenizers, an src and a tgt tokenizer.
# We need to return both of them, but auto tokenizer does not support 2
# tokenizers, so the first one is used. This is fine, unless you are
# training/setting/encoding the prompt of the decoder in the target language, 
# which will fail ugly.
def get_marian_fast_tokenizers(tokenizer) -> Tokenizer:
    from transformers.models.marian.tokenization_marian import MarianTokenizer

    tokenizer: MarianTokenizer = tokenizer

    src_tokenizer = MarianConverter(tokenizer, "encoder").converted()
    src_tokenizer.enable_padding(
        pad_id=tokenizer.pad_token_id, pad_token=tokenizer.pad_token
    )
    tgt_tokenizer = MarianConverter(tokenizer, "decoder").converted()
    tgt_tokenizer.enable_padding(
        pad_id=tokenizer.pad_token_id, pad_token=tokenizer.pad_token
    )
    # get the tokenizer type:
    assert src_tokenizer.get_vocab() == tgt_tokenizer.get_vocab()
    assert tokenizer.get_vocab() == src_tokenizer.get_vocab()

    return src_tokenizer, tgt_tokenizer


def get_fast_tokenizer(tokenizer) -> Tokenizer:
    if isinstance(tokenizer, MarianTokenizer):
        return get_marian_fast_tokenizers(tokenizer)[0]
    else:
        raise ValueError(f"Unsupported tokenizer type: {type(tokenizer)}")
