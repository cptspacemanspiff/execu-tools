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
#include <executorch/runtime/core/tag.h>

/**
 * @brief Converts a Tag enum to a string.
 *
 */
using executorch::runtime::Tag;
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

std::string to_string(const executorch::runtime::Tag &tag, int _) {
  return tagToString(tag);
}


using executorch::runtime::Error;
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
std::string to_string(const executorch::runtime::Error &tag, int _) {
  return errorToString(tag);
}

