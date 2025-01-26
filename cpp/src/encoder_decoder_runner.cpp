#include <ExecuTools/encoder_decoder_runner.h>
#include <cstddef>
#include <cstdint>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/tensor_shape_dynamism.h>
#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/log.h>
#include <memory>
#include <string>
#include <string_view>
#include <tokenizers_cpp.h>
#include <vector>

using namespace executools;

// using namespace executorch::runtime;

EncoderDecoderRunner::EncoderDecoderRunner(
    const std::string &model_path,
    executorch::extension::Module::LoadMode load_mode,
    std::unique_ptr<executorch::runtime::EventTracer> event_tracer)
    : MultiEntryPointRunner(model_path, load_mode, std::move(event_tracer)) {

  initialize_tokenizer();
}

void EncoderDecoderRunner::set_decoder_callback(
    std::function<void(const std::vector<std::string> &)> decoder_callback) {
  this->decoder_callback_ = decoder_callback;
}

executorch::runtime::Error
EncoderDecoderRunner::run(const std::vector<std::string> &input_strings) {

  // encode the input strings:
  auto [tokens, mask] = tokenizer_->EncodeBatchWithMask(input_strings, true);
  ET_LOG(Info, "Encoded %zu strings, with a length of %zu.", tokens.size(),
         tokens[0].size());

  int batch_size = tokens.size();
  int seq_len = tokens[0].size();

  using TokenType = int32_t;
  auto kTokenType = executorch::aten::ScalarType::Int;

  // // construct the input to the encoder:
  auto encoder_input =
      executorch::extension::empty({batch_size, seq_len}, kTokenType,executorch::runtime::TensorShapeDynamism::DYNAMIC_UNBOUND);
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

  // TODO get the token from the model.
  auto prefill_prompt = executorch::extension::empty({1, 1}, kTokenType,executorch::runtime::TensorShapeDynamism::DYNAMIC_UNBOUND);
  auto uint32_ptr = prefill_prompt->mutable_data_ptr<TokenType>();
  uint32_ptr[0] = 59513;

  ET_CHECK_OK_OR_RETURN_ERROR(this->module_.load_method("reset_encode_prefill"),
                              "Could not load reset_encoder_prefill method");
  auto encoder_output = ET_UNWRAP(
      this->module_.execute("reset_encode_prefill",
                            {encoder_input, encoder_mask, prefill_prompt}),
      "Could not execute reset_encode_prefill method");

  // call the decoder callback:
  //   decoder_callback_(decoded_strings);
  return executorch::runtime::Error::Ok;
}

executorch::runtime::Error EncoderDecoderRunner::initialize_tokenizer() {
  // pull the tokenizer json from the model:
  auto method_names = ET_UNWRAP(module_.method_names(),
                                "Could not get method names in tokenizer "
                                "initialization");
  ET_CHECK_OR_RETURN_ERROR(
      method_names.find("tokenizer_blob") != method_names.end(), NotImplemented,
      "tokenizer_blob method not found in model, was it exported?");

  // load the method:
  ET_CHECK_OK_OR_RETURN_ERROR(module_.load_method("tokenizer_blob"),
                              "Found tokenizer_blob method, but could not load "
                              "it in tokenizer initialization");

  // get the tokenizer blob:
  std::vector<executorch::runtime::EValue> tokenizer_blob_tensor =
      ET_UNWRAP(module_.execute("tokenizer_blob"),
                "Could not execute tokenizer_blob method");

  auto tensor = tokenizer_blob_tensor[0].toTensor();
  std::string_view tokenizer_blob_view(
      reinterpret_cast<const char *>(tensor.const_data_ptr()), tensor.nbytes());
  std::string tokenizer_blob_str(tokenizer_blob_view.data(),
                                 tokenizer_blob_view.size());
  this->tokenizer_ = tokenizers::HFTokenizer::FromBlobJSON(tokenizer_blob_str);

  ET_LOG(Info, "Successfully loaded HF Tokenizer blob, beginning with: %s ...",
         tokenizer_blob_str.substr(0, 20).c_str());
  return executorch::runtime::Error::Ok;
}
