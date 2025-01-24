
#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <ExecuTools/multi_entry_point_runner.h>
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
  void set_decoder_callback(std::function<void(std::string)> decoder_callback);

  /**
   * @brief Run the encoder decoder runner.
   *
   * @param input_strings text strings to encode, and then decode, in a batch.
   * tokenization, masking is handled internally.
   */
  void run(const std::vector<std::string> &input_strings);

protected:
  // init tokenizer:
  void initialize_tokenizer();

  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;

  // hf tokenizer json:
  std::vector<uint8_t> hf_tokenizer_json_;
  // decoder callback:
  std::function<void(std::string)> decoder_callback_;
};

} // namespace executools