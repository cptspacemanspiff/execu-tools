#include <cstddef>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/tag.h>
#include <iostream>
#include <ostream>
#include <string>

using namespace executorch::extension;
using namespace executorch::runtime;

#include <iostream>
#include <stdexcept>

#include <iostream>
#include <string>

// prints the error comment from executorch/runtime/core/error.h
std::string getErrorComment(Error error) {
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
#include <typeinfo>
// template <typename T> std::string toString(T value) {
//   auto fallback_name = typeid(value).name();
//   return fallback_name;
// }

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/core/tag.h>

// // Function to map Tag enum to string
std::string toString(Tag tag) {
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

// generic function to convert arbitrary type to string:
template <typename T>
std::string to_string(const T &value, int indent_level = 0) {
  return typeid(value).name();
}

template <> std::string to_string(const Tag &tag, int indent_level) {
  return toString(tag);
}

template <> std::string to_string(const Error &error, int indent_level) {
  return getErrorComment(error);
}


// #include <string>
#include <sstream>

template <>
std::string to_string(const TensorInfo &tensor_info, int indent_level) {
  std::ostringstream oss;

  std::string indent(indent_level * 2, ' ');

  oss << '\n';
  // Convert sizes and dim_order to string representation
  oss << indent << "Sizes: [";
  for (const auto &size : tensor_info.sizes()) {
    oss << size << ", ";
  }
  if (!tensor_info.sizes().empty()) {
    oss.seekp(-2, std::ios_base::end); // Remove the trailing comma and space
  }
  oss << "]\n";

  oss << indent << "Dimension Order: [";
  for (const auto &order : tensor_info.dim_order()) {
    oss << static_cast<int>(order) << ", ";
  }
  if (!tensor_info.dim_order().empty()) {
    oss.seekp(-2, std::ios_base::end); // Remove the trailing comma and space
  }
  oss << "]\n";

  // Add other properties
  oss << indent << "Scalar Type: " << to_string(tensor_info.scalar_type())
      << "\n";
  oss << indent << "Is Memory Planned: "
      << (tensor_info.is_memory_planned() ? "true" : "false") << "\n";
  oss << indent << "Size in Bytes: " << tensor_info.nbytes() << "\n";

  return oss.str();
}

// Assume Result<T> has a method `value()` to get the underlying value or
// `has_error()` to check for errors
template <typename T>
void PrintResult(const Result<T> &result, int indent_level = 0) {
  if (!result.ok()) {
    auto error_message = to_string(result.error());
    std::cout << "Error: " << error_message << "\n";
  } else {
    T value = result.get();
    std::cout << "" << to_string(value, indent_level) << "\n";
  }
}

void PrintMethodMeta(const MethodMeta &meta, int indent_level = 0) {
  try {
    std::string indent(indent_level * 2, ' ');

    std::cout <<indent<< "Method Name: " << meta.name() << "\n";
    std::cout <<indent<< "Number of Inputs: " << meta.num_inputs() << "\n";

    for (size_t i = 0; i < meta.num_inputs(); ++i) {
      std::cout <<indent<< "  Input " << i << " Tag: ";
      PrintResult(meta.input_tag(i), indent_level + 1);

      auto tag = meta.input_tag(i);
      if (tag.ok() && tag.get() == Tag::Tensor) {
        std::cout << indent<<"  Input " << i << " Tensor Meta: ";
        PrintResult(meta.input_tensor_meta(i), indent_level+2);
      }
    }

    std::cout  <<indent<< "Number of Outputs: " << meta.num_outputs() << "\n";

    for (size_t i = 0; i < meta.num_outputs(); ++i) {
      std::cout <<indent<< "  Output " << i << " Tag: ";
      PrintResult(meta.output_tag(i),indent_level + 1);

      auto tag = meta.output_tag(i);
      if (tag.ok() && tag.get() == Tag::Tensor) {
        std::cout <<indent<< "  Output " << i << " Tensor Meta: ";
        PrintResult(meta.output_tensor_meta(i), indent_level + 2);
      }
    }

    std::cout  <<indent<< "Number of Memory-Planned Buffers: "
              << meta.num_memory_planned_buffers() << "\n";

    for (size_t i = 0; i < meta.num_memory_planned_buffers(); ++i) {
      std::cout <<indent<< "  Buffer " << i << " Size: ";
      PrintResult(meta.memory_planned_buffer_size(i), indent_level+2);
    }
  } catch (const std::exception &e) {
    std::cerr << "Exception occurred while printing MethodMeta: " << e.what()
              << "\n";
  }
}

void method_infos(Module &module) {
  auto method_names = module.method_names();
  for (const auto &method : *method_names) {
    std::cout << "method name '" << method
              << "' is loaded: " << module.is_method_loaded(method)
              << std::endl;

    std::cout << "loading method '" << method << "'" << std::endl;
    auto valid = module.load_method(method);
    if (valid == executorch::runtime::Error::Ok) {
      std::cout << "method name '" << method
                << "' successfully loaded: " << module.is_method_loaded(method)
                << std::endl;
      const auto method_meta = module.method_meta(method);
      PrintMethodMeta(method_meta.get(),1);
    } else {
      std::cout << "method name '" << method << "' failed to load" << std::endl;
    }
    std::cout << std::endl;
    // print out the method metadata internals:
  }
}

int main() {
  Module module(
      "/home/nlong/StudioProjects/UT/app/src/main/cpp/UTNativeWhisper/python/"
      "execu-tools/python/tests/export_artifacts/stateful_model.pte");

  method_infos(module);

  std::cout << "Module loaded" << std::endl;
}
