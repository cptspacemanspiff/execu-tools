#include "constant_method_utils.h"
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
#include <tokenizers_cpp.h>
#include <vector>

using namespace executools;

// using namespace executorch::runtime;

EncoderDecoderRunner::EncoderDecoderRunner(
    const std::string &model_path,
    executorch::extension::Module::LoadMode load_mode,
    std::unique_ptr<executorch::runtime::EventTracer> event_tracer)
    : MultiEntryPointRunner(model_path, load_mode, std::move(event_tracer)) {

  this->event_tracer()->set_event_tracer_debug_level(
      executorch::runtime::EventTracerDebugLogLevel::kProgramOutputs);
  this->event_tracer()->set_event_tracer_profiling_level(
      executorch::runtime::EventTracerProfilingLevel::kProfileAllEvents);

  // initialize the tokenizer:
  auto tokenizer_blob_tensor = this->execute("tokenizer_blob", {});
  ET_CHECK_MSG(tokenizer_blob_tensor.ok() == 1,
               "Could not execute tokenizer_blob method, was it exported?");
  std::string tokenizer_blob_str =
      constant_method_utils::tensor_byte_blob_to_string(
          executorch::extension::TensorPtr(
              &tokenizer_blob_tensor.get()[0].toTensor()));
  this->tokenizer_ = std::make_unique<HFStringTokenizer>(tokenizer_blob_str);
}

void EncoderDecoderRunner::set_decoder_callback(
    std::function<void(const std::vector<std::string> &)> decoder_callback) {
  this->decoder_callback_ = decoder_callback;
}

executorch::runtime::Result<std::vector<std::string>>
EncoderDecoderRunner::run(const std::vector<std::string> &input_strings) {
  // run the init to zero out data: (probably not needed.)
  auto et_init_method = ET_UNWRAP(this->execute("et_module_init", {}),
                                  "Could not execute et_module_init method");

  // run the encoder + prefill:
  auto [encoder_input, encoder_mask] =
      ET_UNWRAP(this->tokenizer_->strings_to_tensors(input_strings),
                "Could not convert strings to tensors");

  ET_LOG(Info, "Encoded %zu strings, with a length of %zu.",
         encoder_input->size(0), encoder_input->size(1));

  // TODO get the token from the model.
  auto prefill_prompt = executorch::extension::empty(
      {1}, executorch::runtime::etensor::ScalarType::Int,
      executorch::runtime::TensorShapeDynamism::DYNAMIC_UNBOUND);
  auto uint32_ptr = prefill_prompt->mutable_data_ptr<int32_t>();
  uint32_ptr[0] = 59513;

  auto encoder_output =
      ET_UNWRAP(this->execute("reset_encode_prefill",
                              {encoder_input, encoder_mask, prefill_prompt}),
                "Could not execute reset_encode_prefill method");

  auto finished = encoder_output[0].toTensor().const_data_ptr<bool>();
  auto new_tokens = encoder_output[1].toTensor();
  auto past_decoder_outputs =
      executorch::extension::clone_tensor_ptr(encoder_output[2].toTensor());

  // write the new tokens to the decoder callback:
  this->decoder_callback_(this->tokenizer_->tensors_to_strings(
      executorch::extension::make_tensor_ptr(new_tokens)));

  while (finished[0] != true) {
    // call the decoder:
    auto decoder_output =
        ET_UNWRAP(this->execute("decode", {encoder_input, encoder_mask,
                                           past_decoder_outputs}),
                  "Could not execute decode method");
    finished = decoder_output[0].toTensor().const_data_ptr<bool>();
    new_tokens = decoder_output[1].toTensor();
    past_decoder_outputs =
        executorch::extension::clone_tensor_ptr(decoder_output[2].toTensor());

    // write the new tokens to the decoder callback:
    this->decoder_callback_(this->tokenizer_->tensors_to_strings(
        executorch::extension::make_tensor_ptr(new_tokens)));
  }

  return this->tokenizer_->tensors_to_strings(past_decoder_outputs);
}
