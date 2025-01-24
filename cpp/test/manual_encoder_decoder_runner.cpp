#include <CLI/CLI.hpp>
#include <ExecuTools/encoder_decoder_runner.h>
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
      nullptr);

  // Run the runner
//   runner.run(input_strings);

  return 0;
}