/**
 * @file string_helpers.cpp
 * @author Nicholas Long
 * @brief Helper funtions to convert executorch types to strings.
 * @version 0.1
 * @date 2025-01-03
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <ExecuTools/utils/string_helpers.h>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/tag.h>

#include <sstream>

namespace executools {
namespace utils {

using executorch::runtime::Error;
using executorch::runtime::Tag;

/**
 * @brief Converts a Tag enum to a string.
 *
 */
std::string tagToString(Tag tag) {
  switch (tag) {
#define CASE_TAG(x)                                                            \
  case Tag::x:                                                                 \
    return #x;
    EXECUTORCH_FORALL_TAGS(CASE_TAG)
#undef CASE_TAG
  default:
    return "Unknown";
  }
}

template <> std::string to_string(const executorch::runtime::Tag &tag, int _) {
  return tagToString(tag);
}

/**
 * @brief Converts Result Error Codes to strings:
 *
 */
// prints the error comment from executorch/runtime/core/error.h
std::string errorToString(Error error) {
  switch (error) {
  // System errors
  case Error::Ok:
    return "Status indicating a successful operation.";
  case Error::Internal:
    return "An internal error occurred.";
  case Error::InvalidState:
    return "Status indicating the executor is in an invalid state for a target "
           "operation.";
  case Error::EndOfMethod:
    return "Status indicating there are no more steps of execution to run.";

  // Logical errors
  case Error::NotSupported:
    return "Operation is not supported in the current context.";
  case Error::NotImplemented:
    return "Operation is not yet implemented.";
  case Error::InvalidArgument:
    return "User provided an invalid argument.";
  case Error::InvalidType:
    return "Object is an invalid type for the operation.";
  case Error::OperatorMissing:
    return "Operator(s) missing in the operator registry.";

  // Resource errors
  case Error::NotFound:
    return "Requested resource could not be found.";
  case Error::MemoryAllocationFailed:
    return "Could not allocate the requested memory.";
  case Error::AccessFailed:
    return "Could not access a resource.";
  case Error::InvalidProgram:
    return "Error caused by the contents of a program.";

  // Delegate errors
  case Error::DelegateInvalidCompatibility:
    return "Init stage: Backend receives an incompatible delegate version.";
  case Error::DelegateMemoryAllocationFailed:
    return "Init stage: Backend fails to allocate memory.";
  case Error::DelegateInvalidHandle:
    return "Execute stage: The handle is invalid.";

  // Default case for unknown errors
  default:
    return "Unknown error.";
  }
}

template <>
std::string to_string(const executorch::runtime::Error &tag, int _) {
  return errorToString(tag);
}


std::string get_indent(int indent_level) {
  return std::string(indent_level * 2, ' ');
}

template <>
std::string to_string(const executorch::runtime::TensorInfo &tensor_info,
                      int indent_level) {
  std::ostringstream oss;
  std::string indent = get_indent(indent_level);

  oss << '\n';
  // Convert sizes and dim_order to string representation
  oss << indent << "Sizes: [";
  for (const auto &size : tensor_info.sizes()) {
    oss << size << ", ";
  }
  if (!tensor_info.sizes().empty()) {
    oss.seekp(-2, std::ios_base::end); // Remove trailing comma and space
  }
  oss << "]\n";

  oss << indent << "Dimension Order: [";
  for (const auto &order : tensor_info.dim_order()) {
    oss << static_cast<int>(order) << ", ";
  }
  if (!tensor_info.dim_order().empty()) {
    oss.seekp(-2, std::ios_base::end); // Remove trailing comma and space
  }
  oss << "]\n";

  // Add other properties
  // oss << indent << "Scalar Type: " << tensor_info.scalar_type() << "\n";
  oss << indent << "Is Memory Planned: "
      << (tensor_info.is_memory_planned() ? "true" : "false") << "\n";
  oss << indent << "Size in Bytes: " << tensor_info.nbytes() << "\n";

  return oss.str();
}

template <>
std::string to_string(const executorch::runtime::MethodMeta &method_meta,
                      int indent_level) {
  std::ostringstream oss;
  std::string indent = get_indent(indent_level);

  oss << '\n';
  oss << indent << "Number of Inputs: " << method_meta.num_inputs() << "\n";
  oss << indent << "Number of Outputs: " << method_meta.num_outputs() << "\n";

  oss << indent << "Inputs: \n";
  indent_level++;

  // Print input tags
  for (size_t i = 0; i < method_meta.num_inputs(); ++i) {
    indent = get_indent(indent_level);
    oss << indent << "Input " << i << " Tag: ";
    auto tag = method_meta.input_tag(i);
    oss << resultToString(tag, indent_level) << "\n";
    if (tag.ok()) {
      // Print tensor meta if input is a tensor
      if (tag.get() == executorch::runtime::Tag::Tensor) {
        oss << indent << "Input " << i << " Tensor Meta: "
            << resultToString(method_meta.input_tensor_meta(i),
                              indent_level + 1);
      }
    }
  }

  indent_level--;
  indent = get_indent(indent_level);
  oss << indent << "Outputs: \n";
  indent_level++;

  // Print output tags
  for (size_t i = 0; i < method_meta.num_outputs(); ++i) {
    indent = get_indent(indent_level);
    oss << indent << "Output " << i << " Tag: ";
    auto tag = method_meta.output_tag(i);
    oss << resultToString(tag, indent_level) << "\n";
    if (tag.ok()) {
      // Print tensor meta if output is a tensor
      if (tag.get() == executorch::runtime::Tag::Tensor) {
        oss << indent << "Output " << i << " Tensor Meta: "
            << resultToString(method_meta.output_tensor_meta(i),
                              indent_level + 1);
      }
    }
  }

  indent_level--;
  indent = get_indent(indent_level);

  // Print memory planned buffers
  oss << indent << "Number of Memory-Planned Buffers: "
      << method_meta.num_memory_planned_buffers() << "\n";

  for (size_t i = 0; i < method_meta.num_memory_planned_buffers(); ++i) {
    oss << indent << "Buffer " << i << " Size: ";
    auto buffer_size = method_meta.memory_planned_buffer_size(i);
    if (buffer_size.ok()) {
      oss << buffer_size.get() << "\n";
    } else {
      oss << "Error: " << errorToString(buffer_size.error()) << "\n";
    }
  }

  return oss.str();
}

} // namespace utils
} // namespace executools
