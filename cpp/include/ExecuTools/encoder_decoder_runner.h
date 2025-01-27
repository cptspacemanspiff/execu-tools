
#pragma once

#include <ExecuTools/string_tokenizer.h>
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
  executorch::runtime::Result<std::vector<std::string>>
  run(const std::vector<std::string> &input_strings);

protected:
  // decoder callback:
  std::function<void(const std::vector<std::string> &)> decoder_callback_;

private:
  // string tokenizer:
  std::unique_ptr<HFStringTokenizer> tokenizer_;
};

} // namespace executools