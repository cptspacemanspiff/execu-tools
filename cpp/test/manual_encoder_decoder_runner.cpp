#include <CLI/CLI.hpp>
#include <ExecuTools/encoder_decoder_runner.h>
#include <executorch/devtools/etdump/etdump_flatcc.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/assert.h>
#include <memory>
#include <string>
#include <vector>

using namespace executools;

int main(int argc, char **argv) {
  CLI::App app{"Encoder Decoder Runner"};

  // Define variables to store the parsed values
  std::vector<std::string> input_strings;
  std::string model_path;

  // Add options to the parser
  app.add_option("-m,--model", model_path, "Path to the model file")
      ->required();
  app.add_option("-i,--input", input_strings, "Input strings to process")
      ->required()
      ->expected(1, -1); // Accept 1 or more inputs

  // Parse the command line
  CLI11_PARSE(app, argc, argv);
  EncoderDecoderRunner runner(
      model_path, executorch::extension::Module::LoadMode::MmapUseMlock,
      std::make_unique<executorch::etdump::ETDumpGen>());

  // create a decoder callback lambda:
  auto decoder_callback =
      [](const std::vector<std::string> &new_token_strings) {
        std::cout << new_token_strings[0]; // TODO: print all of them
      };

  runner.set_decoder_callback(decoder_callback);

  std::cout << "Running runner, with input strings: " << std::endl;
  for (const auto &input_string : input_strings) {
    std::cout << "  '" << input_string << "'" << std::endl;
  }

  // Run the runner
  auto maybe_result = runner.run(input_strings);
  // write out the et_dump

  auto buffer = runner.get_event_tracer_dump();

  // write out the et_dump
  // grab the directory from the model path and filename without extension
  auto dir = std::filesystem::path(model_path).parent_path();
  auto file_name = std::filesystem::path(model_path).stem();
  auto et_dump_path = dir / (file_name.string() + ".etdump");

  std::ofstream ofs(et_dump_path.string(), std::ios::out | std::ios::binary);
  ofs.write(reinterpret_cast<const char *>(buffer.data()), buffer.size());
  ofs.close();


  ET_CHECK_MSG(maybe_result == executorch::runtime::Error::Ok,
               "\n    Decoder runner run, ran into something.");

  // torch::executor::etdump_result result = runner.save_event_tracer_dump();


  return 0;
}