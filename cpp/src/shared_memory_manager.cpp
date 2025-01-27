#include "ExecuTools/shared_memory_manager.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/result.h>
#include <map>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include <executorch/runtime/core/hierarchical_allocator.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/executor/method_meta.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/log.h>

#include "constant_method_utils.h"
#include "reserved_fn_names.h"

using namespace executools;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;

SharedMemoryManager::SharedMemoryManager(
    std::shared_ptr<Program> program,
    executorch::runtime::EventTracer *event_tracer)
    : program_(program),
      init_method_(reserved_fn_names::SHARED_BUFFER_INIT_FN) {
  // get all methods in the program:
  auto num_methods = program_->num_methods();
  std::unordered_map<std::string, executorch::runtime::MethodMeta>
      method_meta_map;
  method_meta_map.reserve(num_methods);
  for (size_t i = 0; i < num_methods; ++i) {
    auto method_name = program_->get_method_name(i);
    // TODO: assert(method_name.ok());
    auto method_meta = program_->method_meta(method_name.get());
    // TODO: assert(method_meta.ok());
    method_meta_map.emplace(
        std::make_pair(method_name.get(), method_meta.get()));
  }
  // get the shared memory buffer names/ids (this runs constant methods):
  this->shared_memory_plan_map_ =
      get_shared_memory_plan_map(method_meta_map, event_tracer);
  // from the memory plan map, get a vector of the shared memory ids:
  std::vector<int64_t> shared_memory_ids;
  for (const auto &plan : this->shared_memory_plan_map_) {
    // the memory plan is 1 indexed, so we need to subtract 1 to get the actual
    // mem_id which is the python size -1 b/c of wierd reserve rules.
    int64_t actual_mem_id = plan.second.mem_id - 1;
    // if the mem_id is not in the shared_memory_ids_ vector, add it:
    if (std::find(shared_memory_ids.begin(), shared_memory_ids.end(),
                  actual_mem_id) == shared_memory_ids.end()) {
      shared_memory_ids.push_back(actual_mem_id);
    }
  }

  this->shared_memory_ids_ = shared_memory_ids;

  // Check that the in init_method_ is in the method_meta_map:
  auto it = method_meta_map.find(init_method_);
  ET_CHECK_MSG(it != method_meta_map.end(),
               "Init method %s not found, required for SharedMemoryManager",
               init_method_.c_str()); // TODO: Death Test

  // now that we have the init method, we allocate memory for it first (it has
  // a complete view of the shared memory buffer.):
  allocate_memory_for_method(it->first, it->second);

  for (const auto &method_pair : method_meta_map) {
    if (method_pair.first !=
        init_method_) { // skip init method, already allocated.
      allocate_memory_for_method(method_pair.first, method_pair.second);
    }
  }

  // save the the shared buffer views to internal member (for debugging):
  this->shared_buffer_views_ = get_shared_buffer_views();
}

std::shared_ptr<HierarchicalAllocator>
SharedMemoryManager::get_allocator(const std::string &method_name) {
  auto it = method_allocator_map_.find(method_name);
  if (it == method_allocator_map_.end()) {
    ET_CHECK_MSG(false,
                 "Allocator for method not found in SharedMemoryManager");
  }
  return it->second;
}

void SharedMemoryManager::allocate_memory_for_method(
    const std::string &method_name,
    const executorch::runtime::MethodMeta &method_meta) {
  using executorch::runtime::Span;
  // vector for normal buffers (owns the memory):
  {
    std::vector<std::unique_ptr<uint8_t[]>> normal_buffers;

    // vector of arenas (does not own the memory, but needs to be kept alive):
    std::vector<Span<uint8_t>> arenas;

    size_t num_memory_planned_buffers =
        method_meta.num_memory_planned_buffers();
    for (size_t id = 0; id < num_memory_planned_buffers; ++id) {
      size_t buffer_size =
          static_cast<size_t>(method_meta.memory_planned_buffer_size(id).get());

      // for each regular id we create a buffer, and add it to the arena.
      // if the id is not in the shared_memory_ids_ vector, it is a normal
      // buffer.
      if (std::find(shared_memory_ids_.begin(), shared_memory_ids_.end(), id) ==
          shared_memory_ids_.end()) {
        normal_buffers.emplace_back(std::make_unique<uint8_t[]>(buffer_size));
        arenas.emplace_back(
            Span<uint8_t>({normal_buffers.back().get(), buffer_size}));
        ET_LOG(Info,
               "Allocated normal buffer for method %s, memory id %zu, size %zu",
               method_name.c_str(), id, buffer_size);
      } else {
        // this is a shared memory id:
        // check if the buffer is already in the shared_memory_buffers_ map:
        auto it = shared_memory_buffers_.find(id);
        if (it == shared_memory_buffers_.end()) {
          // it is not, create a new buffer:
          auto buffer = std::make_unique<uint8_t[]>(buffer_size);
          shared_memory_buffers_[id] =
              std::make_pair(std::move(buffer), buffer_size);
          ET_LOG(Info,
                 "Allocated shared buffer for method %s, memory id %zu, "
                 "size %zu",
                 method_name.c_str(), id, buffer_size);
        } else {
          // the buffer is already in the map, so we don't need to create a
          // new one. first validate that the buffer size is the same as the
          // one we are trying to allocate:
          ET_CHECK_MSG(shared_memory_buffers_[id].second == buffer_size,
                       "Buffer size mismatch for already "
                       "allocated shared memory id");
          ET_LOG(Info,
                 "Reusing shared buffer for method %s, memory id %zu, "
                 "size %zu",
                 method_name.c_str(), id, buffer_size);
        }
        // in both cases, add the buffer to the arena.
        arenas.emplace_back(
            Span<uint8_t>({shared_memory_buffers_[id].first.get(),
                           shared_memory_buffers_[id].second}));
      }
    }

    // we now have the arenas and normal buffers (holding non-shared
    // memory).place them into a method storage object, and save as private
    // member.
    MethodDataStore method_data_store{std::move(arenas),
                                      std::move(normal_buffers)};
    auto pair = method_data_store_map_.emplace(
        std::make_pair(method_name, std::move(method_data_store)));
    if (!pair.second) {
      ET_CHECK_MSG(false, "Method data store already exists for method %s",
                   method_name.c_str());
    }
  }
  // this is stupid, lookup the arenas we just put in, but we just moved alot
  // of stuff w/ unique_ptr so some things are invalidated. dont you love c++,
  // footguns abound.
  auto arenas_data = method_data_store_map_.at(method_name).arenas.data();
  auto arenas_size = method_data_store_map_.at(method_name).arenas.size();
  // now that the memory is maintained, create a heiracacleallocator:

  auto method_allocator_ptr = std::shared_ptr<HierarchicalAllocator>(
      new HierarchicalAllocator({arenas_data, arenas_size}));
  method_allocator_map_.insert(
      std::make_pair(method_name, method_allocator_ptr));
}

std::pair<uint8_t *, size_t>
SharedMemoryManager::get_buffer(const std::string &method_name, size_t mem_id,
                                size_t mem_offset_byte, size_t mem_size_byte) {

  uint8_t *buffer_ptr = nullptr;
  size_t buffer_size = 0;

  // check if the given mem_id is in the shared_memory_ids_ vector:
  if (std::find(shared_memory_ids_.begin(), shared_memory_ids_.end(), mem_id) !=
      shared_memory_ids_.end()) {
    // it is a shared memory id, so we need to return the buffer from the
    // shared_memory_buffers_ map:
    auto it = shared_memory_buffers_.find(mem_id);
    if (it == shared_memory_buffers_.end()) {
      ET_CHECK_MSG(false, "Buffer not found shared memory id %zu", mem_id);
    }
    buffer_ptr = it->second.first.get();
    buffer_size = it->second.second;
  } else {
    // not a shared memory id:
    // it is not a shared memory id, so we need to return the buffer from the
    // method data store:
    auto it = method_data_store_map_.find(method_name);
    if (it == method_data_store_map_.end()) {
      ET_CHECK_MSG(false, "Method data store not found for method %s",
                   method_name.c_str());
    }

    // TODO: allow loading buffers from method data store.
    ET_CHECK_MSG(false,
                 "Could not find mem_id %zu, buffer in shared buffers, "
                 "Loading buffers from method data store not "
                 "supported",
                 mem_id);
  }
  // check that the buffer is in

  ET_CHECK_MSG(buffer_ptr != nullptr, "Buffer pointer is null");

  if (mem_size_byte == 0 & mem_offset_byte == 0) {
    // return the entire buffer:
    return std::make_pair(buffer_ptr, buffer_size);
  }

  // check that the size is not 0:
  ET_CHECK_MSG(mem_size_byte != 0,
               "Buffer size cannot be 0, if mem_offset_byte is not 0 (we "
               "return the whole buffer)");

  // check that the mem_offset_byte + mem_size_byte are within the bounds of
  // the buffer:
  ET_CHECK_MSG(!(mem_offset_byte + mem_size_byte > buffer_size),
               "Buffer offset and size out of bounds for method %s, "
               "mem_id %zu, mem_offset_byte %zu, mem_size_byte %zu, and a "
               "buffer size of %zu",
               method_name.c_str(), mem_id, mem_offset_byte, mem_size_byte,
               buffer_size);

  // return the buffer with the given offset and size:
  return std::make_pair(buffer_ptr + mem_offset_byte, mem_size_byte);
}

// Private internal helper functions:
std::map<std::string, SharedMemoryManager::MemoryPlanInfo>
SharedMemoryManager::get_shared_memory_plan_map(
    const std::unordered_map<std::string, executorch::runtime::MethodMeta>
        &method_meta_map,
    executorch::runtime::EventTracer *event_tracer) const {

  // check that the method_meta_map has the required methods:
  auto it = method_meta_map.find(reserved_fn_names::GET_SHARED_BUFFER_NAMES_FN);
  ET_CHECK_MSG(it != method_meta_map.end(),
               "Method %s not found in method_meta_map",
               reserved_fn_names::GET_SHARED_BUFFER_NAMES_FN);
  it = method_meta_map.find(reserved_fn_names::GET_SHARED_BUFFER_MEMORY_PLAN_FN);
  ET_CHECK_MSG(it != method_meta_map.end(),
               "Method %s not found in method_meta_map",
               reserved_fn_names::GET_SHARED_BUFFER_MEMORY_PLAN_FN);

  // get the buffer names:
  auto buffer_names_tensor =
      constant_method_utils::execute_constant_method_with_temp_memory(
          program_.get(), event_tracer,
          reserved_fn_names::GET_SHARED_BUFFER_NAMES_FN,
          method_meta_map.at(reserved_fn_names::GET_SHARED_BUFFER_NAMES_FN));
  auto buffer_names =
      constant_method_utils::tensor_cstr_to_string(buffer_names_tensor);

  // get the memory plan:
  auto memory_plan_tensor =
      constant_method_utils::execute_constant_method_with_temp_memory(
          program_.get(), event_tracer,
          reserved_fn_names::GET_SHARED_BUFFER_MEMORY_PLAN_FN,
          method_meta_map.at(
              reserved_fn_names::GET_SHARED_BUFFER_MEMORY_PLAN_FN));

  ET_CHECK_MSG(
      memory_plan_tensor->dtype() == executorch::aten::ScalarType::Long,
      "Memory plan tensor dtype is not executorch::aten::ScalarType::Long");
  using memory_plan_dtype = int64_t;

  // ordered map is easier to debug.
  std::map<std::string, MemoryPlanInfo> shared_memory_plan_map;
  auto memory_plan_data =
      memory_plan_tensor->const_data_ptr<memory_plan_dtype>();
  for (int64_t i = 0; i < memory_plan_tensor->size(0); ++i) {
    auto plan_ptr = memory_plan_data + i * memory_plan_tensor->size(1);
    MemoryPlanInfo memory_plan_info{
        plan_ptr[0], plan_ptr[1], plan_ptr[2],
        plan_ptr[3], plan_ptr[4], plan_ptr[5],
    };

    shared_memory_plan_map.insert(
        std::make_pair(buffer_names[i], memory_plan_info));
  }

  return shared_memory_plan_map;
}

std::map<std::string, SharedMemoryManager::SharedBufferView>
SharedMemoryManager::get_shared_buffer_views() {
  const auto &shared_memory_plan_map = this->shared_memory_plan_map_;

  std::map<std::string, SharedBufferView> shared_buffer_views;
  for (const auto &plan : shared_memory_plan_map) {
    auto buffer_name = plan.first;
    auto buffer_plan = plan.second;

    auto actual_mem_id = buffer_plan.mem_id - 1;

    auto [buffer, buffer_size] = this->get_buffer(
        reserved_fn_names::SHARED_BUFFER_INIT_FN, actual_mem_id,
        (size_t)buffer_plan.mem_offset, (size_t)buffer_plan.mem_actual_size);

    auto view = std::span<uint8_t>(buffer, buffer_size);
    shared_buffer_views.emplace(std::make_pair(buffer_name, std::move(view)));
  }
  return shared_buffer_views;
}
