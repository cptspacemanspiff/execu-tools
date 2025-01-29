# Overview:
>[!NOTE]
> This is my understanding after just about a month of looking at this, so there is a high possibility that my understanding is incorrect. I would appreciate feedback.

When using executor to run models models with state, it can become painful. The biggest issue that I have encountered was that unlike torch.compile torch.export does not allow (easily) for data-dependent control flow.

Because of this, even models that run with torch.compile in strict mode can be hard to get working in torch.export/executorch.

My solution to this is to be able to call multiple torch.module methods and have them share state. Then we get behavior most like torch compile/ the python torch.

## API:

To do this I created a `MultiEntryPointExporter` class that manages the export process, the api for its use is below:

### Python Export:

```python
class StatefulModel(torch.nn.Module):
    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.register_buffer(
            "cache",
            torch.zeros((max_batch_size, max_seq_len), dtype=torch.float32),
            persistent=True,
        )    

    # need slicing here:
    def set_cache(self, data: torch.Tensor):
        self.cache[0 : data.shape[0], 0 : data.shape[1]] = data
        return None

    # need narrow here:
    def get_cache(self, data: torch.Tensor):
        narrowed_cache = self.cache.narrow(0, 0, data.size(0)).narrow(1, 0, data.size(1))
        data.copy_(narrowed_cache)
        return None

def test_stateful_export():
    max_batch_size = 10
    max_seq_len = 20

    # Wrap/Create a model to export. The model CANNOT have a forward method.
    model = StatefulModel(max_batch_size=max_batch_size, max_seq_len=max_seq_len)

    # Exporter class that manages the legwork of 
    exporter = MultiEntryPointExporter(model)

    # Register the buffer by fqn 
    # Alternatively pass a module fqn, which will register every registered buffer inside it.
    exporter.register_shared_buffer("cache")

    # Define dynamic dimensions as normal
    batch_size = Dim("batch_size_dim", min=1, max=max_batch_size)
    seq_len = Dim("seq_len_dim", min=1, max=max_seq_len)

    # Register methods for export, with examples for tracing.
    exporter.register(
        model.set_cache,
        data=MethodArg(
            torch.ones(max_batch_size-1, max_seq_len-1),
            dynamic_dims={0: batch_size, 1: seq_len},
        ),
    )

    exporter.register(
        model.get_cache,
        data=MethodArg(
            torch.ones(max_batch_size-1, max_seq_len-1),
            dynamic_dims={0: batch_size, 1: seq_len},
        ),
    )

    # Pass additional data to the runtime.
    constant_methods = {'my_const_function':torch.zeros(3,3)}

    # Export process 
    # I have not yet played with quantization or backends, there should not be issues.
    # I hope...
    exporter.export()
    exporter.to_edge(constant_methods=constant_methods)
    exporter.to_executorch()
    exporter.save(output_dir, "stateful_model") # Also saves a ton of diagnostic info.
```

### C++ Runtime:
```c++
#include "ExecuTools/shared_memory_manager.h"
#include "ExecuToolsTestDirs.h"
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/platform/log.h>
using executorch::extension::Module;
using executorch::runtime::Error;
using executorch::runtime::Program;

ET_NODISCARD Error run_program() {
  // create a module:
  Module MultiEntryModule(EXECUTOOLS_PYTHON_ARTIFACT_DIR
                          "/StatefulModel/stateful_model.pte",
                          Module::LoadMode::MmapUseMlock, nullptr);
  // force load the program:
  ET_CHECK_OK_OR_RETURN_ERROR(MultiEntryModule.load(), "Failed to load module");
  auto program = MultiEntryModule.program();
  // validate that the program is loaded:
  ET_CHECK_OR_RETURN_ERROR(program != nullptr, InvalidProgram,
                           "Program is not loaded");

  // use the shared_ptr program to construct a shared memory manager:
  executools::SharedMemoryManager shared_memory_manager(program);

  ET_CHECK_OK_OR_RETURN_ERROR(
      MultiEntryModule.load_method(
          "set_cache", nullptr,
          shared_memory_manager.get_allocator("set_cache").get()),
      "Failed to load set_cache");
  ET_CHECK_OK_OR_RETURN_ERROR(
      MultiEntryModule.load_method(
          "get_cache", nullptr,
          shared_memory_manager.get_allocator("get_cache").get()),
      "Failed to load get_cache");

  const int batch_size = 10;
  const int seq_len = 20;

  // lambda function to check if the two tensors are the same:
  auto tensors_equal = [](const executorch::extension::TensorPtr &t1,
                          const executorch::extension::TensorPtr &t2,
                          size_t size) {
    auto ptr1 = t1->const_data_ptr<float>();
    auto ptr2 = t2->const_data_ptr<float>();
    for (size_t i = 0; i < size; i++) {
      if (ptr1[i] != ptr2[i]) {
        return false;
      }
    }
    return true;
  };

  auto set_input = executorch::extension::ones({batch_size, seq_len});
  auto get_input = executorch::extension::zeros({batch_size, seq_len});
  // the tensors are not equal.
  if (tensors_equal(set_input, get_input, batch_size * seq_len)) {
    return Error::InvalidState;
  }
  // Run the model, set the cache with the value from the input.
  auto none_result_1 =
      ET_UNWRAP(MultiEntryModule.execute("set_cache", set_input));
  // Run the model, get the cache, is returned into the 
  auto none_result_2 =
      ET_UNWRAP(MultiEntryModule.execute("get_cache", get_input));

  // Get input has now been filled with ones that were set into the cache.
  if (!tensors_equal(set_input, get_input, batch_size * seq_len)) {
    return Error::InvalidState;
  }
  return Error::Ok;
}

int main() {
  if (run_program() != Error::Ok) {
    ET_LOG(Error, "Test failed");
    return 1;
  }
  ET_LOG(Info, "Test passed");
  return 0;
}
```

## How does this work (High-level):

the main steps in the process are: 

1. MultiEntryPointExporter gets set by the user with data on what to export, this includes shared buffers and module methods.
2. MultiEntryPointExporter monkey patches the forward method, (this seems to be an undocumented capability, but works reliably so far...)
3. Create a synthetic method that just mutates the values of all the shared buffers, this gives us a method to base our shared memory plan on.
4. Run torch.export to export the model.
5. On the resulting graph, for every shared buffer in the method, after the placeholders we inject a self.shared_buffer.copy_(self.shared_buffer). This forces the buffer to be treated as mutable, even if the method just reads it.
6. We run to_edge.
7. Since at this point all shared buffers are registered as mutable in the graph signature, we remove any copy operations where both the source and target are the same and pointing to a shared buffer. (This removes the op that we added earlier, during to_edge it may have gotten removed, but in the case where there is a later mutation, it would not have been.)
9. We run our memory planning via to_executorch on our synthetic buffer init method, placing all shared buffers in mem_id 2. This mem_id 2 plan is saved for future reference.
10. we rerun the memory planning on all methods, but after the planner has run, we overwrite the memory location for all objects in mem_id 2 to use the reference memory plan. This ensures that all objects are placed at the same offsets in the buffer.## related pull requests

A more in depth writeup on the export process on rational/why I did it is here: [Design Reasoning](DesignReasoning.md)

## Additional Stuff:

The exporter also does a few other things:

* It passes the shared buffer memory plan to the runtime via 2 constant methods, one contains a tensor of c-string buffer names and alongside each buffer's memory plan. This makes it easy to view debug info on the runtime side, though currently I have only used it in the debugger. It also allows the runtime to dynamically identify the memory ID
* It auto saves a ton of data for the export process, including etrecord and memory debugging plan.

## Sharp Edges, My personal to-do list, bugs I still need to create tests/issues for:

### Todo:

* Currently have not dug into the quantization or tested this running on a backend.
* The current memory planner on the c++ side places each method's non shared memory in a method specific buffer. This is less than Ideal because it means ram usage grows with the number of methods exported.
  * On the export side I need to add a check that all method state buffers are marked as shared, then pass a flag to the runtime, to say 'reuse the largest buffer for all planned methods.' This should be safe b/c we cannot call methods at the same time anyway.
  * in the future it might be useful to auto-detect these buffers, which would probably make auto exporting more models easier.
* The inspector for etdumps/etrecord support is not great. (I made changes to make it work but there is weird behavior.)
* I want to integrate this w/ the bundled program stuff so that we can get validation testing at the same time.
* The module export wrappers are a work in progress, I have really only exported 1, the opus model. My goal is to generally have wrappers that map to hugging face model types.
  * I figure as I export more models, the process will gradually get more general, as I lift more things that are model specific into the wrapper initialization logic.
  * also need to have automated unit tests for the models in the wrapper so that validating python-python behavior is consistant.
* I am using a hacked in argmax for greedy search, ideally I would like to use huggingface search implementations directly (sampling, beam, etc.) I have not yet started investigating what needs to be done in order to get them export compatable, and am also unsure if the methods required are in core aten (ie torch.multinomial for sampling). But it would be awesome to the sampler and beam search decoding working in this so nothing needs to be rewritten in c++.  
* This is relying on my branch of Transformers, executorch, and tokenizers-cpp, I need to finish my pull-requests and get stuff merged (I had been waiting to make sure that my understanding/proof of concept worked)
* add more unit tests.
* do a search through my code and do everything marked `todo`

### Sharp edges:

* might be obvious, but this is not thread safe.
* While I tried to not use any explicit dynamic memory allocations on the runtime side, I am using std::vector all over the place, so so much for that.
* The method of adding a "stupid" copy operation to ensure that a buffer is marked mutable might be fragile. In an ideal world that would be optimized out, the fact that it is not might be considered a bug. 

### Bugs (still need to dig into/ add an issue): 

* torch.export dynamic shapes infer constant if the example to trace is the same size as the dynamic dimensions max, (or min?). I throw a error in this lib if you do this, but it should be fixed/error out in torch.export.
* optimized kernels segfault if batch=2 in the translation example. have not looked yet.

>[!TIP]
>If you read this far, I am just finishing off taking for a year long travel sabbatical, am currently unemployed and am actively looking for work.
>
>I started down this rabbit hole b/c I wanted to do on-device AI for a side-project, to get more experience in the types of projects I want to work on and to have something to show off. 
>Also building things is way more fun and interesting than grinding leetcode and throwing my resume into the ether when applying for jobs. 
>
>If you found it useful, are hiring (or know someone), I would love to work on this type of thing and get paid for it.
>
>My objective is to build cool AI related stuff with awesome people, ideally in a fast-paced environment while having a blast doing it. I currently live on the east coast, but am willing to relocate to most anywhere... but not Texas.
>
>[LinkedIn](https://www.linkedin.com/in/nicholas-long-z42/)