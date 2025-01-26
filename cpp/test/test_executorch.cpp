
#include "executorch/extension/data_loader/file_data_loader.h"
#include <ExecuTools/utils/string_helpers.h>
#include <cassert>
#include <cstddef>
#include <executorch/devtools/etdump/etdump_flatcc.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <executorch/runtime/core/result.h>
#include <executorch/runtime/executor/program.h>
#include <iostream>

#include "ExecuToolsTestDirs.h"
#include <ExecuTools/shared_memory_manager.h>

#include <executorch/runtime/core/error.h>
#include <fstream>
#include <memory>
#include <span>
using executorch::extension::FileDataLoader;
using executorch::extension::Module;
using executorch::runtime::Error;
using executorch::runtime::Program;
using executorch::runtime::Result;

ET_NODISCARD Error run_program() {

  // hardcoded values:
  std::string init_method = "et_module_init";
  std::size_t shared_memory_id = 1; // one less than the python side.

  auto etdump_gen_original = std::make_unique<executorch::etdump::ETDumpGen>();
  // create a module:
  Module MultiEntryModule(EXECUTOOLS_PYTHON_ARTIFACT_DIR "/StatefulModel/stateful_model.pte",
                          Module::LoadMode::MmapUseMlock,
                          std::move(etdump_gen_original));
  // force load the program:
  ET_CHECK_OK_OR_RETURN_ERROR(MultiEntryModule.load(), "Failed to load module");
  auto program = MultiEntryModule.program();
  // validate that the program is loaded:
  ET_CHECK_OR_RETURN_ERROR(program != nullptr, InvalidProgram,
                           "Program is not loaded");

  // use the shared_ptr program to construct a shared memory manager:
  executools::SharedMemoryManager shared_memory_manager(
      program, {shared_memory_id}, init_method);
  // internal memory view for debugging:
  auto shared_memory_info =
      shared_memory_manager.get_buffer<float>(init_method, shared_memory_id);
  std::span<float, std::dynamic_extent> shared_memory_view_span{
      shared_memory_info.first, shared_memory_info.second};

  auto set_cache_info = shared_memory_manager.get_buffer<float>("set_cache", 0);
  std::span<float, std::dynamic_extent> set_cache_view_span{
      set_cache_info.first, set_cache_info.second};

  auto get_cache_info = shared_memory_manager.get_buffer<float>("get_cache", 0);
  std::span<float, std::dynamic_extent> get_cache_view_span{
      get_cache_info.first, get_cache_info.second};

  ET_CHECK_OK_OR_RETURN_ERROR(
      MultiEntryModule.load_method(
          init_method, nullptr,
          shared_memory_manager.get_allocator(init_method).get()),
      "Failed to load init_method: %s", init_method.c_str());
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

  ET_UNWRAP(MultiEntryModule.execute(init_method),
            "Failed to execute et_module_init");
  // TODO: dynamic shapes are broken...

  // method_infos(module);
  const int batch_size = 10;
  const int seq_len = 20;
  // Wrap the input data with a Tensor.
  float input[batch_size * seq_len];
  auto tensor = executorch::extension::from_blob(input, {
                                                            2,
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
                                                              1,
                                                              10,
                                                          });
  auto result2 = MultiEntryModule.execute("get_cache", tensor2);

  if (result2.ok()) {
    auto outputs = executools::utils::resultToString(result2);
    std::cout << "Success: " << outputs << std::endl;
  } else {
    std::cout << "Error: " << executools::utils::to_string(result2.error())
              << std::endl;
  }

  {
    auto event_tracer = MultiEntryModule.event_tracer();
    auto et_out = dynamic_cast<executorch::etdump::ETDumpGen *>(event_tracer);
    auto buffer = et_out->get_etdump_data();

    // Create a new vector with the buffer data
    auto size = buffer.size;
    std::vector<uint8_t> buffer_vector(size);
    std::copy_n(static_cast<const uint8_t *>(buffer.buf), size,
                buffer_vector.data());
    // write the etdump to a file:
    std::ofstream ofs(EXECUTOOLS_PYTHON_ARTIFACT_DIR
                      "/StatefulModel/stateful_model.etdump",
                      std::ios::out | std::ios::binary);
    ofs.write(reinterpret_cast<const char *>(buffer_vector.data()),
              buffer_vector.size());
    ofs.close();
  }

  return Error::Ok;
}

int main() { run_program(); }
