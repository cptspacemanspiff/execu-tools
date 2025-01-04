
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <ExecuTools/utils/string_helpers.h>

#include <iostream>

using executorch::extension::Module;

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
      std::cout << executools::utils::resultToString(method_meta, 1);
    } else {
      std::cout << "method name '" << method << "' failed to load" << std::endl;
    }
    std::cout << std::endl;
    // print out the method metadata internals:
  }
}

int main() {
  Module module(
      "/home/nlong/execu-tools/python/tests/export_artifacts/stateful_model.pte");

  method_infos(module);

  std::cout << "Module loaded" << std::endl;
}
