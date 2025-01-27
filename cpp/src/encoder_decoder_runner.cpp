#include <ExecuTools/encoder_decoder_runner.h>
#include <cstdint>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/event_tracer.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/portable_type/scalar_type.h>
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

  this->module_.event_tracer()->set_event_tracer_debug_level(
      executorch::runtime::EventTracerDebugLogLevel::kProgramOutputs);
  this->module_.event_tracer()->set_event_tracer_profiling_level(
      executorch::runtime::EventTracerProfilingLevel::kProfileAllEvents);

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

  // load the methods TODO: (move to init section)
  ET_CHECK_OK_OR_RETURN_ERROR(this->module_.load_method("et_module_init"),
                              "Could not load et_module_init method");
  ET_CHECK_OK_OR_RETURN_ERROR(this->module_.load_method("reset_encode_prefill"),
                              "Could not load reset_encoder_prefill method");
  ET_CHECK_OK_OR_RETURN_ERROR(this->module_.load_method("decode"),
                              "Could not load decode method");

  // run the init to zero out data: (probably not needed.)
  auto et_init_method = ET_UNWRAP(this->module_.execute("et_module_init"),
                                  "Could not execute et_module_init method");

  // run the encoder + prefill:
  auto [encoder_input, encoder_mask] =
      ET_UNWRAP(strings_to_tensors(input_strings),
                "Could not convert strings to tensors");

  // TODO get the token from the model.
  auto prefill_prompt = executorch::extension::empty(
      {1}, executorch::runtime::etensor::ScalarType::Int,
      executorch::runtime::TensorShapeDynamism::DYNAMIC_UNBOUND);
  auto uint32_ptr = prefill_prompt->mutable_data_ptr<int32_t>();
  uint32_ptr[0] = 59513;

  auto encoder_output = ET_UNWRAP(
      this->module_.execute("reset_encode_prefill",
                            {encoder_input, encoder_mask, prefill_prompt}),
      "Could not execute reset_encode_prefill method");

  auto finished = encoder_output[0].toTensor().const_data_ptr<bool>();
  auto new_tokens = encoder_output[1].toTensor();
  auto past_decoder_outputs =
      executorch::extension::clone_tensor_ptr(encoder_output[2].toTensor());

  // write the new tokens to the decoder callback:
  this->decoder_callback_(tensors_to_strings(new_tokens));

  int i = 0;
  // while (finished[0] != true) {
  while (i < 5) {
    // call the decoder:
    auto decoder_output = ET_UNWRAP(
        this->module_.execute("decode",
                              {encoder_input, encoder_mask,
                               past_decoder_outputs}),
        "Could not execute decode method");
    finished = decoder_output[0].toTensor().const_data_ptr<bool>();
    new_tokens = decoder_output[1].toTensor();
    past_decoder_outputs =
        executorch::extension::clone_tensor_ptr(decoder_output[2].toTensor());

    // write the new tokens to the decoder callback:
    this->decoder_callback_(tensors_to_strings(new_tokens));
    i++;
  }

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

executorch::runtime::Result<std::pair<executorch::extension::TensorPtr,
                                      executorch::extension::TensorPtr>>
EncoderDecoderRunner::strings_to_tensors(
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

std::vector<std::string> EncoderDecoderRunner::tensors_to_strings(
    const executorch::runtime::etensor::Tensor &tensor_ptr) {
  auto data = tensor_ptr.const_data_ptr<int32_t>();
  std::vector<std::string> decoded_strings;

  // todo dont use a for loop, construct the vector directly from data.
  int batch_size = tensor_ptr.size(0);
  for (int i = 0; i < batch_size; i++) {
    std::vector<int> decoded_tokens_for_sequence;
    for (int j = 0; j < tensor_ptr.size(1); j++) {
      decoded_tokens_for_sequence.push_back(data[i * tensor_ptr.size(1) + j]);
    }
    decoded_strings.push_back(
        this->tokenizer_->Decode(decoded_tokens_for_sequence));
  }

  return decoded_strings;
}
