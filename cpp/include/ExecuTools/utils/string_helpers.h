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

} // namespace utils
} // namespace executools