/**
 * @file reserved_fn_names.h
 * @author your name (you@domain.com)
 * @brief Private header for reserved function names.
 * @version 0.1
 * @date 2025-01-27
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

namespace executools {

/**
 * @brief Reserved function names for the executools shared buffers.
 *
 * These are the same (manually updated) as in the
 * python/execu_tools/model_exporter.py file.
 */
namespace reserved_fn_names {
// Common init function for shared buffers:
extern const char *SHARED_BUFFER_INIT_FN;

// Constant methods for shared buffers memory plan.

// A (num shared buffers) x (max cstr len)
// tensor of the cstr shared buffer names (uint8_t).
extern const char *GET_SHARED_BUFFER_NAMES_FN;

// A (num shared buffers) x (6) tensor of the cstr shared buffer memory info.
// [mem_id, mem_obj_id, mem_offset, mem_allocated_size, mem_actual_size,
// num_elements]
extern const char *GET_SHARED_BUFFER_MEMORY_PLAN_FN;

} // namespace reserved_fn_names
} // namespace executools