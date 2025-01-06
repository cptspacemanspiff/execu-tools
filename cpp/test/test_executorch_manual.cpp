// manual instatiation of executorch.

#include "executorch/extension/data_loader/file_data_loader.h"
#include "executorch/extension/memory_allocator/malloc_memory_allocator.h"
#include "executorch/runtime/core/exec_aten/util/scalar_type_util.h"
#include "executorch/runtime/core/result.h"
#include "executorch/runtime/executor/program.h"
#include <cassert>
#include <executorch/runtime/platform/runtime.h>
#include <iostream>
#include <memory>
#include <span>

#include <executorch/devtools/etdump/etdump_flatcc.h>

#include "ExecuToolsTestDirs.h"

int main() {

  executorch::runtime::runtime_init();

  // create a new executorch program.
  using executorch::aten::ScalarType;
  using executorch::aten::Tensor;
  using executorch::aten::TensorImpl;
  using executorch::extension::FileDataLoader;
  using executorch::extension::MallocMemoryAllocator;
  using executorch::runtime::Error;
  using executorch::runtime::EValue;
  using executorch::runtime::HierarchicalAllocator;
  using executorch::runtime::MemoryManager;
  using executorch::runtime::Method;
  using executorch::runtime::MethodMeta;
  using executorch::runtime::Program;
  using executorch::runtime::Result;
  using executorch::runtime::Span;

  Result<FileDataLoader> loader = FileDataLoader::from(
      EXECUTOOLS_PYTHON_ARTIFACT_DIR "/stateful_model.pte");
  assert(loader.ok());

  Result<Program> program = Program::load(&loader.get());
  assert(program.ok());

  // create a new memory manager.
  // Method names map back to Python nn.Module method names. Most users will
  // only
  // have the singular method "forward".
  const char *method_name = "get_cache";

  // MethodMeta is a lightweight structure that lets us gather metadata
  // information about a specific method. In this case we are looking to get the
  // required size of the memory planned buffers for the method "forward".
  Result<MethodMeta> method_meta = program->method_meta(method_name);
  assert(method_meta.ok());

  std::vector<std::unique_ptr<uint8_t[]>> planned_buffers; // Owns the Memory
  std::vector<Span<uint8_t>> planned_arenas; // Passed to the allocator

  size_t num_memory_planned_buffers = method_meta->num_memory_planned_buffers();
  std::cout << "num_memory_planned_buffers: " << num_memory_planned_buffers
            << std::endl;
  // It is possible to have multiple layers in our memory hierarchy; for
  // example, SRAM and DRAM.
  for (size_t id = 0; id < num_memory_planned_buffers; ++id) {
    // .get() will always succeed because id < num_memory_planned_buffers.
    size_t buffer_size =
        static_cast<size_t>(method_meta->memory_planned_buffer_size(id).get());
    planned_buffers.push_back(std::make_unique<uint8_t[]>(buffer_size));
    planned_arenas.push_back({planned_buffers.back().get(), buffer_size});
  }
  HierarchicalAllocator planned_memory(
      {planned_arenas.data(), planned_arenas.size()});

  auto memory_span_0 =
      std::span<float>(reinterpret_cast<float *>(planned_arenas[0].data()),
                       planned_arenas[0].size() / sizeof(float));

  auto memory_span_1 =
      std::span<float>(reinterpret_cast<float *>(planned_arenas[1].data()),
                       planned_arenas[1].size() / sizeof(float));

  // Version of MemoryAllocator that uses malloc to handle allocations rather
  // then a fixed buffer.
  MallocMemoryAllocator method_allocator;

  // Assemble all of the allocators into the MemoryManager that the Executor
  // will use.
  MemoryManager memory_manager(&method_allocator, &planned_memory);


  executorch::etdump::ETDumpGen etdump_gen{};
  
  Result<Method> method =
      program->load_method(method_name, &memory_manager, &etdump_gen);
  assert(method.ok());

  // Create our input tensor.
  float data[10 * 20];
  for (int i = 0; i < 10 * 20; i++) {
    data[i] = i;
  }
  Tensor::SizesType sizes[] = {10, 20};
  Tensor::DimOrderType dim_order[] = {0, 1};
  TensorImpl impl(ScalarType::Float, // dtype
                  2,                 // number of dimensions
                  sizes, data, dim_order);
  Tensor t(&impl);

  // Implicitly casts t to EValue
  Error set_input_error = method->set_input(t, 0);
  assert(set_input_error == Error::Ok);

  Error execute_error = method->execute();
  assert(execute_error == Error::Ok);

  EValue output = method->get_output(0);

  auto output_address = (float *)output.payload.as_tensor.mutable_data_ptr();

  auto data_addr = (float *)data;
  auto memory_span_0_addr = (memory_span_0.data());
  auto memory_span_1_addr = (memory_span_1.data());

  std::cout << "memory_span_0_addr: " << std::hex << memory_span_0_addr
            << std::endl;
  std::cout << "memory_span_1_addr: " << std::hex << memory_span_1_addr
            << std::endl;
  std::cout << "data_addr: " << std::hex << data_addr << std::endl;
  std::cout << "actual output_address: " << std::hex << output_address
            << std::endl;

  std::cout << "output: " << std::endl;
  torch::executor::etdump_result result = etdump_gen.get_etdump_data();
  if (result.buf != nullptr && result.size > 0) {
    // On a device with a file system users can just write it out
    // to the file-system.
    FILE *f = fopen("etdump.bin", "w+");
    fwrite((uint8_t *)result.buf, 1, result.size, f);
    fclose(f);
    free(result.buf);
  }

  std::cout << "output: done" << std::endl; 
}