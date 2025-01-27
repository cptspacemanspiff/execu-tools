#include "constant_method_utils.h"
#include <cstdint>
#include <executorch/extension/memory_allocator/malloc_memory_allocator.h>
namespace executools {

namespace constant_method_utils {

executorch::extension::TensorPtr execute_constant_method_with_temp_memory(
    executorch::runtime::Program *program,
    executorch::runtime::EventTracer *event_tracer, std::string method_name,
    const executorch::runtime::MethodMeta &method_meta) {
  // execute the method with temp memory TODO: make this all stack allocated...
  std::vector<uint8_t> temp_memory;

  auto num_memory_planned_buffers = method_meta.num_memory_planned_buffers();

  ET_CHECK_MSG(
      num_memory_planned_buffers == 0,
      "Expected constant method, which has no memory planned buffers.");

  // create a memory allocator:
  executorch::extension::MallocMemoryAllocator method_allocator;
  // create a memory manager, constant method so planned memory is nullptr, and
  // temp memory is nullptr:
  executorch::runtime::MemoryManager memory_manager{&method_allocator, nullptr,
                                                    nullptr};

  // load the method:
  auto method =
      program->load_method(method_name.c_str(), &memory_manager, event_tracer);
  ET_CHECK_MSG(method.ok(), "Constant method %s failed to load",
               method_name.c_str());

  ET_CHECK_MSG(method->inputs_size() == 0,
               "Constant method %s has inputs, expected 0",
               method_name.c_str());
  ET_CHECK_MSG(method->outputs_size() == 1,
               "Constant method %s has outputs, expected 1",
               method_name.c_str());

  // execute the method:
  std::string block_name = "ExecuteConstantMethod::" + method_name;
  std::string profile_event_name = "ExecuteConstantMethod::" + method_name;
  auto success =
      method->execute(block_name.c_str(), profile_event_name.c_str());
  ET_CHECK_MSG(success == executorch::runtime::Error::Ok,
               "Constant method %s failed to execute", method_name.c_str());

  // we do not own the memory of the output tensor, so we need to copy it to a
  // new tensor:
  auto tmp_tensor = method->get_output(0).toTensor();

  auto tensor = executorch::extension::clone_tensor_ptr(tmp_tensor);

  return tensor;
}

std::vector<std::string>
tensor_cstr_to_string(const executorch::extension::TensorPtr &tensor) {

  // make sure that the tensor is 2d:
  ET_CHECK_MSG(tensor->dim() == 2,
               "Tensor is not 2d (num_strings x max_str_len)");

  // get the tensor data:
  auto tensor_data = tensor->const_data_ptr<char>();
  // get the number of strings:
  auto num_strings = tensor->size(0);
  std::vector<std::string> strings{};
  strings.reserve(num_strings);

  for (int64_t i = 0; i < num_strings; ++i) {
    // offset into the tensor data:
    auto str_ptr = tensor_data + i * tensor->size(1);
    int64_t str_length = 0;
    while (str_length < tensor->size(1) && str_ptr[str_length] != '\0') {
      str_length++;
    }
    strings.push_back(std::string(str_ptr, str_length));
  }

  return strings;
}

std::string
tensor_byte_blob_to_string(const executorch::extension::TensorPtr &tensor) {
  // make sure that the tensor is 1d:
  ET_CHECK_MSG(tensor->dim() == 1, "Tensor is not 1d (num_bytes)");

  std::string tokenizer_blob_str(tensor->const_data_ptr<char>(),
                                 tensor->size(0));
  return tokenizer_blob_str;
}

} // namespace constant_method_utils

} // namespace executools
