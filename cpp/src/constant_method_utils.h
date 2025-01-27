/**
 * @file constant_method_utils.h
 * @author Nicholas Long
 * @brief Helper functions to run constant methods with temporary memory before
 * program module api is loaded. Used for getting parameter info on how to
 * construct the shared buffers. Mostly used in shared memory manager, but may
 * be useful elsewhere...
 * @date 2025-01-27
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/executor/program.h>

namespace executools {

namespace constant_method_utils {

/**
 * @brief Execute a constant method with temporary memory.
 *
 * @param program Program to execute the method on.
 * @param method_name Name of the method to execute.
 * @param method_meta Method meta data.
 * @return TensorPtr to the output of the method (owns the data).
 */
executorch::extension::TensorPtr execute_constant_method_with_temp_memory(
    executorch::runtime::Program *program,
    executorch::runtime::EventTracer *event_tracer, std::string method_name,
    const executorch::runtime::MethodMeta &method_meta);

/**
 * @brief Convert a tensor of multipled cstr to a vector of strings
 *
 * @param tensor TensorPtr (num_strings x max_str_len) zero padded, null
 * terminated strings.
 * @return std::vector<std::string>
 */
std::vector<std::string>
tensor_cstr_to_string(const executorch::extension::TensorPtr &tensor);

} // namespace constant_method_utils

} // namespace executools
