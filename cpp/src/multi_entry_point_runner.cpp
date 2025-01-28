#include "ExecuTools/shared_memory_manager.h"
#include <ExecuTools/multi_entry_point_runner.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <executorch/devtools/etdump/etdump_flatcc.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/event_tracer.h>
#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/log.h>
#include <memory>

using namespace executools;

MultiEntryPointRunner::MultiEntryPointRunner(
    const std::string &model_path,
    executorch::extension::Module::LoadMode load_mode,
    std::unique_ptr<executorch::runtime::EventTracer> event_tracer)
    : module_(model_path, load_mode, std::move(event_tracer)) {

  // Abort if fails, b/c inside a constructor.
  ET_LOG(Info, "MultiEntryPointRunner: Initializing executorch program");
  ET_CHECK_MSG(initialize_program() == executorch::runtime::Error::Ok,
               "Failed to initialize program in MultiEntryPointRunner");

  ET_LOG(Info, "MultiEntryPointRunner: Initializing shared memory manager");
  // initailize the shared memory manager: (this allocates the memory for the
  // methods), but methods are still not loaded yet.
  this->shared_memory_manager_ = std::make_unique<SharedMemoryManager>(
      module_.program(), module_.event_tracer());

  ET_LOG(Info, "MultiEntryPointRunner: Loading methods");
  // load the methods:
  ET_CHECK_MSG(load_methods() == executorch::runtime::Error::Ok,
               "Failed to load all methods in MultiEntryPointRunner");

  ET_LOG(Info, "MultiEntryPointRunner: Constructor Complete");
  // all methods are loaded.
}

executorch::runtime::EventTracer *MultiEntryPointRunner::event_tracer() {
  return this->module_.event_tracer();
}

executorch::runtime::Result<std::vector<executorch::runtime::EValue>>
MultiEntryPointRunner::execute(
    const std::string &method_name,
    const std::vector<executorch::runtime::EValue> &input_values) {
  return this->module_.execute(method_name, input_values);
}

executorch::runtime::Result<std::unordered_set<std::string>>
MultiEntryPointRunner::method_names() {
  return this->module_.method_names();
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

// std::vector<uint8_t> MultiEntryPointRunner::get_event_tracer_dump() {
//   executorch::runtime::EventTracer *event_tracer = module_.event_tracer();

//   if (event_tracer == nullptr) {
//     ET_LOG(Error, "MultiEntryPointRunner event tracer was not set (nullptr)");
//     return std::vector<uint8_t>();
//   }

//   auto *et_dump_gen =
//       dynamic_cast<executorch::etdump::ETDumpGen *>(event_tracer);
//   ET_CHECK_MSG(et_dump_gen != nullptr,
//                "Failed to cast event tracer to ETDumpGen");
//   auto buffer = et_dump_gen->get_etdump_data();

//   // Create a new vector with the buffer data
//   auto size = buffer.size;
//   std::vector<uint8_t> result(size);
//   std::copy_n(static_cast<const uint8_t *>(buffer.buf), size, result.data());

//   return result;
// }