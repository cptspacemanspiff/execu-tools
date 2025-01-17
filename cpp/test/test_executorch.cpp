
#include "executorch/extension/data_loader/file_data_loader.h"
#include <ExecuTools/utils/string_helpers.h>
#include <cassert>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <executorch/runtime/executor/program.h>
#include <iostream>

#include "ExecuToolsTestDirs.h"
#include <ExecuTools/shared_memory_manager.h>

using executorch::extension::FileDataLoader;
using executorch::extension::Module;
using executorch::runtime::Program;
using executorch::runtime::Result;

int main() {
  // file data loader:
  // Result<FileDataLoader> loader = FileDataLoader::from(
  //     EXECUTOOLS_PYTHON_ARTIFACT_DIR "/stateful_model.pte");
  // assert(loader.ok());
  // // load the program:
  // Result<Program> program = Program::load(&loader.get());
  // assert(program.ok());

  // create a module:
  Module unused_module(EXECUTOOLS_PYTHON_ARTIFACT_DIR "/stateful_model.pte");
  // get the program:
  auto program = unused_module.program();

  // use the shared_ptr program to construct a shared memory manager:
  executools::SharedMemoryManager shared_memory_manager(program);




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
