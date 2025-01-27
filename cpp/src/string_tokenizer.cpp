#include <ExecuTools/string_tokenizer.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/executor/method.h>

using namespace executools;

HFStringTokenizer::HFStringTokenizer(const std::string &tokenizer_blob_str) {
  ET_LOG(Info, "Loading HF Tokenizer blob, beginning with: %s ...",
         tokenizer_blob_str.substr(0, 20).c_str());
  this->tokenizer_ = tokenizers::HFTokenizer::FromBlobJSON(tokenizer_blob_str);
  ET_LOG(Info, "Successfully loaded HF Tokenizer blob, beginning with: %s ...",
         tokenizer_blob_str.substr(0, 20).c_str());
}

executorch::runtime::Result<std::pair<executorch::extension::TensorPtr,
                                      executorch::extension::TensorPtr>>
HFStringTokenizer::strings_to_tensors(
    const std::vector<std::string> &input_strings) {
  // encode the input strings:
  auto [tokens, mask] = tokenizer_->EncodeBatchWithMask(input_strings, true);
  ET_LOG(Info, "Encoded %zu strings, with a length of %zu.", tokens.size(),
         tokens[0].size());

  // TODO: add checks for size
  int batch_size = tokens.size();
  int seq_len = tokens[0].size();

  using TokenType = int32_t;
  auto kTokenType = executorch::aten::ScalarType::Int;

  // // construct the input to the encoder:
  auto encoder_input = executorch::extension::empty(
      {batch_size, seq_len}, kTokenType,
      executorch::runtime::TensorShapeDynamism::DYNAMIC_UNBOUND);
  auto uint32 = encoder_input->mutable_data_ptr<TokenType>();
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < seq_len; j++) {
      uint32[i * seq_len + j] = tokens[i][j];
    }
  }

  auto encoder_mask = executorch::extension::empty(
      {batch_size, seq_len}, executorch::aten::ScalarType::Int,
      executorch::runtime::TensorShapeDynamism::DYNAMIC_UNBOUND);
  auto bool_ptr = encoder_mask->mutable_data_ptr<int32_t>();
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < seq_len; j++) {
      bool_ptr[i * seq_len + j] = bool(mask[i][j]);
    }
  }

  return std::make_pair(encoder_input, encoder_mask);
}

std::vector<std::string> HFStringTokenizer::tensors_to_strings(
    const executorch::extension::TensorPtr &tensor_ptr,
    bool skip_special_tokens) {
  auto data = tensor_ptr->const_data_ptr<int32_t>();
  std::vector<std::string> decoded_strings;

  // todo dont use a for loop, construct the vector directly from data.
  int batch_size = tensor_ptr->size(0);
  for (int i = 0; i < batch_size; i++) {
    std::vector<int> decoded_tokens_for_sequence;
    for (int j = 0; j < tensor_ptr->size(1); j++) {
      decoded_tokens_for_sequence.push_back(data[i * tensor_ptr->size(1) + j]);
    }
    decoded_strings.push_back(this->tokenizer_->Decode(
        decoded_tokens_for_sequence, skip_special_tokens));
  }

  return decoded_strings;
}
