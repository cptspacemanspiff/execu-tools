
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <ExecuTools/utils/string_helpers.h>

#include <iostream>

#include "ExecuToolsTestDirs.h"

using executorch::extension::Module;

void method_infos(Module &module) {
  auto method_names = module.method_names();
  for (const auto &method : *method_names) {
    std::cout << "method name '" << method
              << "' is loaded: " << module.is_method_loaded(method)
              << std::endl;

    std::cout << "loading method '" << method << "'" << std::endl;
    auto valid = module.load_method(method);
    if (valid == executorch::runtime::Error::Ok) {
      std::cout << "method name '" << method
                << "' successfully loaded: " << module.is_method_loaded(method)
                << std::endl;
      const auto method_meta = module.method_meta(method);
      std::cout << executools::utils::resultToString(method_meta, 1);
    } else {
      std::cout << "method name '" << method << "' failed to load" << std::endl;
    }
    std::cout << std::endl;
    // print out the method metadata internals:
  }
}

int main() {
  Module module(EXECUTOOLS_PYTHON_ARTIFACT_DIR "/stateful_model.pte");

  // method_infos(module);
  const int batch_size = 1;
  const int seq_len = 20;
  // Wrap the input data with a Tensor.
  float input[batch_size * seq_len];
  auto tensor = executorch::extension::from_blob(input, {
                                                            batch_size,
                                                            seq_len,
                                                        });

  std::fill(input, input + batch_size * seq_len, 1.0f);

  module.load_method("set_cache");
  module.load_method("get_cache");
  // Run the model.
  auto result = module.execute("set_cache", tensor);

  if (result.ok()) {
    auto outputs = executools::utils::resultToString(result);
    std::cout << "Success: " << outputs << std::endl;
  } else {
    std::cout << "Error: " << executools::utils::to_string(result.error())
              << std::endl;
  }

  float output[10 * 20];
  std::fill(output, output + 10 * 20, 2.0f);
  auto tensor2 = executorch::extension::from_blob(output, {
                                                              10,
                                                              20,
                                                          });
  auto result2 = module.execute("get_cache", tensor2);

  if (result2.ok()) {
    auto outputs = executools::utils::resultToString(result2);
    std::cout << "Success: " << outputs << std::endl;
  } else {
    std::cout << "Error: " << executools::utils::to_string(result2.error())
              << std::endl;
  }

  return 0;
}
