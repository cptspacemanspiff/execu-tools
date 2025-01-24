#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <ExecuTools/shared_memory_manager.h>
#include <executorch/extension/module/module.h>
namespace executools {

/**
 * @brief Top level class for running models exported with Multiple Entry
 * Points, and shared state. Does all the initialization and setup, but does not
 * handle running the model, order of ops, etc.
 *
 */
class MultiEntryPointRunner {
public:
  MultiEntryPointRunner(
      const std::string &model_path,
      executorch::extension::Module::LoadMode load_mode,
      std::unique_ptr<executorch::runtime::EventTracer> event_tracer);

protected:
  /**
   * @brief Loads a single method into the program.
   *
   * @param method_name Name of the method to load.
   * @param event_tracer Event tracer to use for the method. defaults to nullptr
   * (use the event tracer of the Module)
   * @return executorch::runtime::Error
   */
  executorch::runtime::Error
  load_method(const std::string &method_name,
              executorch::runtime::EventTracer *event_tracer = nullptr);

  /**
   * @brief Loads all the methods in the program, alternatively may be possible
   * to do lazy init via the load_method function.
   *
   * @return executorch::runtime::Error
   */
  executorch::runtime::Error load_methods();

  /**
   * @brief Validates that the method can be called based on the metadata of the
   * method (ie it creates junk values an sends them to the module, basically a
   * check that nothing segfaults or crashes, or errors.)
   * Generally just creates zeros of the all valid shapes and types and sends
   * them in.
   *
   * @param method_name
   * @return executorch::runtime::Error
   */
  executorch::runtime::Error validate_method(const std::string &method_name);

  executorch::extension::Module module_;
  std::unique_ptr<executools::SharedMemoryManager> shared_memory_manager_;
  std::vector<uint8_t> hf_tokenizer_json;

private:
  executorch::runtime::Error initialize_program();

  std::string init_method_name_;
  std::vector<size_t> shared_memory_ids_;
};
} // namespace executools