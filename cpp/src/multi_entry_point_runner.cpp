#include "ExecuTools/shared_memory_manager.h"
#include <ExecuTools/multi_entry_point_runner.h>
#include <algorithm>
#include <cstdint>
#include <executorch/devtools/etdump/etdump_flatcc.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/event_tracer.h>
#include <executorch/runtime/platform/assert.h>
#include <memory>
#include <cstring>

using namespace executools;

MultiEntryPointRunner::MultiEntryPointRunner(
    const std::string &model_path,
    executorch::extension::Module::LoadMode load_mode,
    std::unique_ptr<executorch::runtime::EventTracer> event_tracer)
    : module_(model_path, load_mode, std::move(event_tracer)) {

  // Abort if fails, b/c inside a constructor.
  ET_CHECK_MSG(initialize_program() == executorch::runtime::Error::Ok,
               "Failed to initialize program in MultiEntryPointRunner");

  // program is initialized, now we can use it to get standard info from the
  // model.
  // TODO: Add this to the constructor.
  // this->init_method_name_ = module_.load_method('executools_init_method'); //
  // constant method that gets the string name of the init method, hardcode for
  // now.
  this->init_method_name_ = "et_module_init";
  // Same for shared memory ids, hardcode for now.
  this->shared_memory_ids_ = {1};

  // initailize the shared memory manager: (this allocates the memory for the
  // methods), but methods are still not loaded yet.
  this->shared_memory_manager_ = std::make_unique<SharedMemoryManager>(
      module_.program(), shared_memory_ids_, init_method_name_);
}

executorch::runtime::Error MultiEntryPointRunner::load_method(
    const std::string &method_name,
    executorch::runtime::EventTracer *event_tracer) {
  // uses the event tracer of the module
  ET_CHECK_OK_OR_RETURN_ERROR(
      module_.load_method(
          method_name, event_tracer,
          shared_memory_manager_->get_allocator(method_name).get()),
      "Load_method: Failed to load method: %s", method_name.c_str());
  return executorch::runtime::Error::Ok;
}

executorch::runtime::Error MultiEntryPointRunner::load_methods() {
  auto method_names = ET_UNWRAP(this->module_.method_names());
  for (const auto &method_name : method_names) {
    ET_CHECK_OK_OR_RETURN_ERROR(load_method(method_name, nullptr),
                                "Load_methods: Failed to load method: %s",
                                method_name.c_str());
  }
  return executorch::runtime::Error::Ok;
}

executorch::runtime::Error
MultiEntryPointRunner::validate_method(const std::string &method_name) {
  // ET_CHECK_OK_OR_RETURN_ERROR(
  //     validate_method(method_name),
  //     "Validate_method: Failed to validate method: %s", method_name.c_str());
  // TODO: Implement this.
  return executorch::runtime::Error::Ok;
}

executorch::runtime::Error MultiEntryPointRunner::initialize_program() {
  ET_CHECK_OK_OR_RETURN_ERROR(module_.load(), "Failed to load module");
  auto program = module_.program();
  ET_CHECK_OR_RETURN_ERROR(program != nullptr, InvalidProgram,
                           "Program is not loaded");
  return executorch::runtime::Error::Ok;
}

std::vector<uint8_t> MultiEntryPointRunner::get_event_tracer_dump() {
  executorch::runtime::EventTracer *event_tracer = module_.event_tracer();
  ET_CHECK_MSG(event_tracer != nullptr,
               "MultiEntryPointRunner event tracer was not set (nullptr)");

  auto *et_dump_gen =
      dynamic_cast<executorch::etdump::ETDumpGen *>(event_tracer);
  ET_CHECK_MSG(et_dump_gen != nullptr,
               "Failed to cast event tracer to ETDumpGen");
  auto buffer = et_dump_gen->get_etdump_data();

  // Create a new vector with the buffer data
  auto size = buffer.size;
  std::vector<uint8_t> result(size);
  std::copy_n(static_cast<const uint8_t*>(buffer.buf), size, result.data());

  return result;
}