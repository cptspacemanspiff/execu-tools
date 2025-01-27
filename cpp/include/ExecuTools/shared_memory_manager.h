#pragma once
#include <ExecuTools/ExecuTools_export.h>

#include <cstddef>
#include <cstdint>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/core/hierarchical_allocator.h>
#include <executorch/runtime/executor/method_meta.h>
#include <executorch/runtime/executor/program.h>

#include <map>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace executools {

ExecuTools_EXPORT class SharedMemoryManager {
public:
  SharedMemoryManager(std::shared_ptr<executorch::runtime::Program> program,
                      executorch::runtime::EventTracer *event_tracer = nullptr);

  std::shared_ptr<executorch::runtime::HierarchicalAllocator>
  get_allocator(const std::string &method_name);

  std::pair<uint8_t *, size_t> get_buffer(const std::string &method_name,
                                          size_t mem_id,
                                          size_t mem_offset_byte = 0,
                                          size_t mem_size_byte = 0);

  template <typename T>
  std::pair<T *, size_t> get_buffer(const std::string &method_name,
                                    size_t mem_id) {
    std::pair<uint8_t *, size_t> buffer = get_buffer(method_name, mem_id);
    return std::make_pair(reinterpret_cast<T *>(buffer.first),
                          buffer.second / sizeof(T));
  }

  // conviencence method for those who use the debugger to view the internal
  // memory buffers.
  union SharedBufferView {
    std::span<uint8_t> uint8_span;
    std::span<uint16_t> uint16_span;
    std::span<uint32_t> uint32_span;
    std::span<uint64_t> uint64_span;
    std::span<int8_t> int8_span;
    std::span<int16_t> int16_span;
    std::span<int32_t> int32_span;
    std::span<int64_t> int64_span;
    std::span<float> float_span;
    std::span<double> double_span;
    std::span<bool> bool_span;
  };

  std::map<std::string, SharedBufferView> get_shared_buffer_views();

private:
  class MemoryPlanInfo {
  public:
    const int64_t mem_id;
    const int64_t mem_obj_id;
    const int64_t mem_offset;
    const int64_t mem_allocated_size;
    const int64_t mem_actual_size;
    const int64_t num_elements;
  };

  // Private internal helper functions:
  std::map<std::string, SharedMemoryManager::MemoryPlanInfo>
  get_shared_memory_plan_map(
      const std::unordered_map<std::string, executorch::runtime::MethodMeta>
          &method_meta_map,
      executorch::runtime::EventTracer *event_tracer) const;

  class MethodDataStore {
  public:
    std::vector<executorch::runtime::Span<uint8_t>> arenas;
    std::vector<std::unique_ptr<uint8_t[]>> normal_buffers;
  };

  void allocate_memory_for_method(
      const std::string &method_name,
      const executorch::runtime::MethodMeta &method_meta);

  // Passed through constructor:
  std::shared_ptr<executorch::runtime::Program> program_;
  std::vector<int64_t> shared_memory_ids_;
  std::string init_method_;

  // appended to in allocate_memory_for_method:
  std::unordered_map<uint32_t, std::pair<std::unique_ptr<uint8_t[]>, size_t>>
      shared_memory_buffers_;
  std::unordered_map<std::string, MethodDataStore> method_data_store_map_;
  std::unordered_map<
      std::string, std::shared_ptr<executorch::runtime::HierarchicalAllocator>>
      method_allocator_map_;

  // not really needed, usefull for debug:
  std::map<std::string, MemoryPlanInfo> shared_memory_plan_map_;
  std::map<std::string, SharedBufferView> shared_buffer_views_;
};
} // namespace executools
