
#pragma once

#include <cstddef>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <ExecuTools/multi_entry_point_runner.h>
#include <tokenizers_c.h>
#include <tokenizers_cpp.h>
namespace executools {

ExecuTools_EXPORT class EncoderDecoderRunner : public MultiEntryPointRunner {

public:
  EncoderDecoderRunner(
      const std::string &model_path,
      executorch::extension::Module::LoadMode load_mode,
      std::unique_ptr<executorch::runtime::EventTracer> event_tracer);

  /**
   * @brief Set the decoder callback object that gets called after the decoder
   * runs.
   *
   * @param decoder_callback
   */
  void set_decoder_callback(
      std::function<void(const std::vector<std::string> &)> decoder_callback);

  /**
   * @brief Run the encoder decoder runner.
   *
   * @param input_strings text strings to encode, and then decode, in a batch.
   * tokenization, masking is handled internally.
   */
  executorch::runtime::Error run(const std::vector<std::string> &input_strings);

protected:
  // Everything tokenizer related:
  // init tokenizer:
  executorch::runtime::Error initialize_tokenizer();
  // strings to tensors:
  executorch::runtime::Result<std::pair<executorch::extension::TensorPtr,
                                        executorch::extension::TensorPtr>>
  strings_to_tensors(const std::vector<std::string> &input_strings);
  // tensors to strings:
  std::vector<std::string>
  tensors_to_strings(const executorch::runtime::etensor::Tensor &tensor_ptr);

  // use the HFTokenizer class (want access to the special tokens flag)
  std::unique_ptr<tokenizers::HFTokenizer> tokenizer_;
  // hf tokenizer json:
  std::vector<uint8_t> hf_tokenizer_json_;

  // decoder callback:
  std::function<void(const std::vector<std::string> &)> decoder_callback_;
};

} // namespace executools