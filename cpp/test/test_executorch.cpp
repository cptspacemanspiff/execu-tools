#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/error.h>
#include <iostream>
#include <ostream>

using namespace executorch::extension;

void method_infos(Module &module) {
  auto method_names = module.method_names();
  for (const auto &method : *method_names) {
    std::cout << "method name '" << method
              << "' is loaded: " << module.is_method_loaded(method)
              << std::endl;
    // auto method_meta = module.method_meta(method);
    std::cout << "loading method '" << method << "'" << std::endl;
    auto valid = module.load_method(method);
    if (valid == executorch::runtime::Error::Ok) {
      std::cout << "method name '" << method
                << "' successfully loaded: " << module.is_method_loaded(method)
                << std::endl;
    } else {
      std::cout << "method name '" << method << "' failed to load" << std::endl;
    }
  }
}

int main() {
  Module module(
      "/home/nlong/StudioProjects/UT/app/src/main/cpp/UTNativeWhisper/python/"
      "execu-tools/python/tests/export_artifacts/stateful_model.pte");

  method_infos(module);

  std::cout << "Module loaded" << std::endl;
}
