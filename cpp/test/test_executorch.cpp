
#include "executorch/extension/data_loader/file_data_loader.h"
#include <ExecuTools/utils/string_helpers.h>
#include <cassert>
#include <cstddef>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <executorch/runtime/core/result.h>
#include <executorch/runtime/executor/program.h>
#include <iostream>

#include "ExecuToolsTestDirs.h"
#include <ExecuTools/shared_memory_manager.h>

#include <executorch/runtime/core/error.h>

using executorch::extension::FileDataLoader;
using executorch::extension::Module;
using executorch::runtime::Error;
using executorch::runtime::Program;
using executorch::runtime::Result;

ET_NODISCARD Error run_program() {

  //hardcoded values:
  std::string init_method = "et_module_init";
  std::size_t shared_memory_id = 2;
  
  // create a module:
  Module MultiEntryModule(EXECUTOOLS_PYTHON_ARTIFACT_DIR "/stateful_model.pte");
  // force load the program:
  ET_CHECK_OK_OR_RETURN_ERROR(MultiEntryModule.load(), "Failed to load module");
  auto program = MultiEntryModule.program();
  // validate that the program is loaded:
  ET_CHECK_OR_RETURN_ERROR(program != nullptr, InvalidProgram,
                           "Program is not loaded");

  // use the shared_ptr program to construct a shared memory manager:
  executools::SharedMemoryManager shared_memory_manager(program, {shared_memory_id}, init_method);

  auto allocator = shared_memory_manager.get_allocator("set_cache");

  ET_CHECK_OK_OR_RETURN_ERROR(
      MultiEntryModule.load_method("et_module_init", nullptr, allocator.get()),
      "Failed to load et_module_init");
  ET_CHECK_OK_OR_RETURN_ERROR(
      MultiEntryModule.load_method("set_cache", nullptr, allocator.get()),
      "Failed to load set_cache");
  ET_CHECK_OK_OR_RETURN_ERROR(
      MultiEntryModule.load_method("get_cache", nullptr, allocator.get()),
      "Failed to load get_cache");
  // TODO: dynamic shapes are broken...

  // method_infos(module);
  const int batch_size = 10;
  const int seq_len = 20;
  // Wrap the input data with a Tensor.
  float input[batch_size * seq_len];
  auto tensor = executorch::extension::from_blob(input, {
                                                            3,
                                                            4,
                                                        });

  std::fill(input, input + batch_size * seq_len, 1.0f);


  // Run the model.
  auto result = MultiEntryModule.execute("set_cache", tensor);

  if (result.ok()) {
    auto outputs = executools::utils::resultToString(result);
    std::cout << "Success: " << outputs << std::endl;
  } else {
    std::cout << "Error: " << executools::utils::to_string(result.error())
              << std::endl;
  }

  float output[10 * 20];
  std::fill(output, output + 10 * 20, -2.0f);
  auto tensor2 = executorch::extension::from_blob(output, {
                                                              5,
                                                              7,
                                                          });
  auto result2 = MultiEntryModule.execute("get_cache", tensor2);

  if (result2.ok()) {
    auto outputs = executools::utils::resultToString(result2);
    std::cout << "Success: " << outputs << std::endl;
  } else {
    std::cout << "Error: " << executools::utils::to_string(result2.error())
              << std::endl;
  }
  return Error::Ok;
}

int main() { run_program(); }
