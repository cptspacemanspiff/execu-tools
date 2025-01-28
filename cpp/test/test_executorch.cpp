
#include "ExecuTools/shared_memory_manager.h"
#include "ExecuToolsTestDirs.h"
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/platform/log.h>
using executorch::extension::Module;
using executorch::runtime::Error;
using executorch::runtime::Program;

ET_NODISCARD Error run_program() {
  // create a module:
  Module MultiEntryModule(EXECUTOOLS_PYTHON_ARTIFACT_DIR
                          "/StatefulModel/stateful_model.pte",
                          Module::LoadMode::MmapUseMlock, nullptr);
  // force load the program:
  ET_CHECK_OK_OR_RETURN_ERROR(MultiEntryModule.load(), "Failed to load module");
  auto program = MultiEntryModule.program();
  // validate that the program is loaded:
  ET_CHECK_OR_RETURN_ERROR(program != nullptr, InvalidProgram,
                           "Program is not loaded");

  // use the shared_ptr program to construct a shared memory manager:
  executools::SharedMemoryManager shared_memory_manager(program);

  ET_CHECK_OK_OR_RETURN_ERROR(
      MultiEntryModule.load_method(
          "set_cache", nullptr,
          shared_memory_manager.get_allocator("set_cache").get()),
      "Failed to load set_cache");
  ET_CHECK_OK_OR_RETURN_ERROR(
      MultiEntryModule.load_method(
          "get_cache", nullptr,
          shared_memory_manager.get_allocator("get_cache").get()),
      "Failed to load get_cache");

  const int batch_size = 10;
  const int seq_len = 20;

  // lambda function to check if the two tensors are the same:
  auto tensors_equal = [](const executorch::extension::TensorPtr &t1,
                          const executorch::extension::TensorPtr &t2,
                          size_t size) {
    auto ptr1 = t1->const_data_ptr<float>();
    auto ptr2 = t2->const_data_ptr<float>();
    for (size_t i = 0; i < size; i++) {
      if (ptr1[i] != ptr2[i]) {
        return false;
      }
    }
    return true;
  };

  auto set_input = executorch::extension::ones({batch_size, seq_len});
  auto get_input = executorch::extension::zeros({batch_size, seq_len});
  if (tensors_equal(set_input, get_input, batch_size * seq_len)) {
    return Error::InvalidState;
  }
  // Run the model, set the cache.
  auto none_result_1 =
      ET_UNWRAP(MultiEntryModule.execute("set_cache", set_input));
  // Run the model, get the cache.
  auto none_result_2 =
      ET_UNWRAP(MultiEntryModule.execute("get_cache", get_input));

  // the contents of the cache should be the same as the set_input (this will
  // fail if)
  if (!tensors_equal(set_input, get_input, batch_size * seq_len)) {
    return Error::InvalidState;
  }
  return Error::Ok;
}

int main() {
  if (run_program() != Error::Ok) {
    ET_LOG(Error, "Test failed");
    return 1;
  }
  ET_LOG(Info, "Test passed");
  return 0;
}
