#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <iostream>

using namespace executorch::extension;

int main() {
  Module module(
      "/home/nlong/StudioProjects/UT/app/src/main/cpp/UTNativeWhisper/python/"
      "execu-tools/python/tests/export_artifacts/stateful_model.pte");

  auto method_names = module.method_names();
  for (const auto& method : *method_names) {
    std::cout << "method name " << method << std::endl;
  }

  std::cout << "Module loaded" << std::endl;
}
