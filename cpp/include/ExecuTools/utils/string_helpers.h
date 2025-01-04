#pragma once // Add this to prevent multiple inclusion

#include <executorch/runtime/core/result.h>
#include <sstream>
#include <string>
#include <typeinfo>

// executorch includes:
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/tag.h>
#include <executorch/runtime/executor/method_meta.h>

namespace executools {
namespace utils {

// generic function to convert arbitrary type to string:
template <typename T>
std::string to_string(const T &value, int _ = 0 /*indent_level*/) {
  return typeid(value).name();
}

// type-specific implementations (in cpp file to avoid header collisions):
template <> std::string to_string(const executorch::runtime::Tag &tag, int _);
template <>
std::string to_string(const executorch::runtime::Error &error, int _);

template <>
std::string to_string(const executorch::runtime::TensorInfo &tensor_info,
                      int indent_level);

template <>
std::string to_string(const executorch::runtime::MethodMeta &method_meta,
                      int indent_level);

// Helper function to print Result<T>
template <typename T>
std::string resultToString(const executorch::runtime::Result<T> &result,
                           int indent_level = 0) {
  if (!result.ok()) {
    auto error_message = to_string(result.error());
    std::ostringstream oss;
    oss << "Error: " << error_message;
    return oss.str();
  } else {
    T value = result.get();
    std::ostringstream oss;
    oss << to_string(value, indent_level);
    return oss.str();
  }
}

} // namespace utils
} // namespace executools
