#pragma once
#include <ExecuTools/ExecuTools_export.h>

#include <cstddef>
#include <cstdint>
#include <executorch/runtime/core/hierarchical_allocator.h>
#include <executorch/runtime/executor/method_meta.h>
#include <executorch/runtime/executor/program.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace executools {

ExecuTools_EXPORT class SharedMemoryManager {
public:
  SharedMemoryManager(std::shared_ptr<executorch::runtime::Program> program,
                      std::vector<size_t> shared_mem_ids={1},// TODO: get this from program (passed throgh constant methods?).
                      std::string init_method = {"et_module_init"}
                      );

  std::shared_ptr<executorch::runtime::HierarchicalAllocator>
  get_allocator(const std::string &method_name);

  std::pair<uint8_t*, size_t> get_buffer(const std::string &method_name, size_t mem_id);

  template<typename T>
  std::pair<T*, size_t> get_buffer(const std::string &method_name, size_t mem_id){
    std::pair<uint8_t*, size_t> buffer = get_buffer(method_name, mem_id);
    return std::make_pair(reinterpret_cast<T*>(buffer.first), buffer.second/sizeof(T));
  }
private:
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
  std::vector<size_t> shared_memory_ids_;
  std::string init_method_;

  //generated in constructor:
  std::unordered_map<std::string, executorch::runtime::MethodMeta>
      method_meta_map_;

  // appended to in allocate_memory_for_method:
  std::unordered_map<uint32_t, std::pair<std::unique_ptr<uint8_t[]>, size_t>>
      shared_memory_buffers_;
  std::unordered_map<std::string, MethodDataStore> method_data_store_map_;
  std::unordered_map<
      std::string, std::shared_ptr<executorch::runtime::HierarchicalAllocator>>
      method_allocator_map_;
};
} // namespace executools
