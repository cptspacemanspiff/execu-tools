
#include "executorch/extension/data_loader/file_data_loader.h"
#include <ExecuTools/utils/string_helpers.h>
#include <cassert>
#include <cmath>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/executor/program.h>
#include <iostream>

#include "ExecuToolsTestDirs.h"
#include <ExecuTools/shared_memory_manager.h>

using executorch::extension::FileDataLoader;
using executorch::extension::Module;
using executorch::runtime::Error;
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
  Module module(EXECUTOOLS_PYTHON_ARTIFACT_DIR "/stateful_model.pte");
  Error err = module.load();
  assert(err == Error::Ok);
  // get the program:
  auto program = module.program();

  // use the shared_ptr program to construct a shared memory manager:
  auto shared_memory_manager = executools::SharedMemoryManager(program, {1});

  // load all methods
  auto methods = module.method_names();
  assert(methods.ok());
  for (const auto &method_name : methods.get()) {
    auto allocator_ptr = shared_memory_manager.get_allocator(method_name);
    auto valid = module.load_method(method_name, nullptr, allocator_ptr.get());
    assert(valid == Error::Ok);
  }

  // method_infos(module);
  const int batch_size = 10;
  const int seq_len = 20;
  // Wrap the input data with a Tensor.
  float input[batch_size * seq_len];
  auto t_input = executorch::extension::from_blob(input, {
                                                             batch_size,
                                                             seq_len,
                                                         });

  std::fill(input, input + batch_size * seq_len, 42.0f);

  // show that the buffers changed appropriatly:
  auto shared_buffer_value_initial_set_cache =
      (*(float *)shared_memory_manager.get_allocator("set_cache")
            .get()
            ->get_offset_address(1, 0, 4)
            .get());
  auto shared_buffer_value_initial_get_cache =
      (*(float *)shared_memory_manager.get_allocator("get_cache")
            .get()
            ->get_offset_address(1, 0, 4)
            .get());

  std::cout << "initial state: " << std::endl;
  std::cout << "  input to set_cache: " << input[0] << std::endl;
  std::cout << "  shared buffer (set_cache): "
            << shared_buffer_value_initial_set_cache << std::endl;
  std::cout << "  shared buffer (get_cache): "
            << shared_buffer_value_initial_get_cache << std::endl;
  // Run the model.
  auto r_set_cache = module.execute("set_cache", t_input);

  auto shared_buffer_value_after_set_cache =
      (*(float *)shared_memory_manager.get_allocator("set_cache")
            .get()
            ->get_offset_address(1, 0, 4)
            .get());
  auto shared_buffer_value_after_get_cache =
      (*(float *)shared_memory_manager.get_allocator("get_cache")
            .get()
            ->get_offset_address(1, 0, 4)
            .get());

  std::cout << "after set_cache: " << std::endl;
  std::cout << "  input to set_cache: " << input[0] << std::endl;
  std::cout << "  shared buffer (set_cache): "
            << shared_buffer_value_after_set_cache << std::endl;
  std::cout << "  shared buffer (get_cache): "
            << shared_buffer_value_after_get_cache << std::endl;

  if (r_set_cache.ok()) {
    auto outputs = executools::utils::resultToString(r_set_cache);
    std::cout << "Success: ran set_cache " << outputs << std::endl;
  } else {
    std::cout << "Error: " << executools::utils::to_string(r_set_cache.error())
              << std::endl;
  }

  float output[batch_size * seq_len];
  std::fill(output, output + batch_size * seq_len, 2.0f);
  auto t_output = executorch::extension::from_blob(output, {
                                                               batch_size,
                                                               seq_len,
                                                           });

  std::cout << "before get_cache: " << std::endl;
  std::cout << "  input to get_cache: " << output[0] << std::endl;
  std::cout << "  shared buffer (set_cache): "
            << shared_buffer_value_after_set_cache << std::endl;
  std::cout << "  shared buffer (get_cache): "
            << shared_buffer_value_after_get_cache << std::endl;

  auto r_get_cache = module.execute("get_cache", t_output);

  auto shared_buffer_value_after_set_cache_2 =
      (*(float *)shared_memory_manager.get_allocator("set_cache")
            .get()
            ->get_offset_address(1, 0, 4)
            .get());
  auto shared_buffer_value_after_get_cache_2 =
      (*(float *)shared_memory_manager.get_allocator("get_cache")
            .get()
            ->get_offset_address(1, 0, 4)
            .get());

  std::cout << "after get_cache (recieves output via input mutation): " << std::endl;
  std::cout << "  input to get_cache: " << output[0] << std::endl;
  std::cout << "  shared buffer (set_cache): "
            << shared_buffer_value_after_set_cache_2 << std::endl;
  std::cout << "  shared buffer (get_cache): "
            << shared_buffer_value_after_get_cache_2 << std::endl;

  if (r_get_cache.ok()) {
    auto outputs = executools::utils::resultToString(r_get_cache);
    std::cout << "Success: ran get_cache " << outputs << std::endl;
  } else {
    std::cout << "Error: " << executools::utils::to_string(r_get_cache.error())
              << std::endl;
  }

  // show that the buffers changed appropriatly:

  return 0;
}
