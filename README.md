# Overview:
>[!NOTE]
> This is my understanding after just about a month of looking at this, so there is a high possibility that my understanding is incorrect. I would appreciate feedback.

When using executor to run models models with state, it can become painful. The biggest issue that I have encountered was that unlike torch.compile torch.export does not allow (easily) for data-dependent control flow.

Because of this, even models that run with torch.compile in strict mode can be hard to get working in torch.export/executorch.

My solution to this is to be able to call multiple torch.module methods and have them share state. Then we get behavior most like torch compile/ the python torch.

## Quick-Start (tested on ubuntu 22.04):

This has alot of my personal branches, but to build it run these commands:

>[!NOTE]
>make sure rust is installed first, and ninja.
>rust is for huggingface tokenizers, used in the runtime build of the translation example, ninja is because I use `-G Ninja` in the setup script (delete if you don't have ninja.).

It uses a unholy mixture of cmake's fetch content and submodules:
* fetchcontent
  * executorch
  * CMakeTools (personal cmake build library, because writing good cmake is a pain in the ***)
  * CLI11
  * tokenizer-cpp (for hugging face tokenizer wrapper+hugging face rust lib),
* and manual bash (for my branch of transformers)

1. creates a python venv. 
2. It runs the cmake configure fetching dependencies.
3. builds/installs executorch python lib.
4. it then installs executools python component
5. it installs my branch of transformers.
6. it then runs the executools c++ project build, rebuilding executorch in the process.

```bash
git clone git@github.com:cptspacemanspiff/execu-tools.git
cd execu-tools
git submodule update --init --recursive
./setup.sh
```

### Examples(from the project root):
#### StatefulModel:

Export the program:

Creates artifacts in: python/tests/export_artifacts/StatefulModel
```bash
python python/tests/manual_test_model_export.py 
```

run the c++ code.

```bash
./cpp/build/stage/bin/ExecuToolsModuleTest
```


#### OPUS en-fr translation:

Export the program:

Creates artifacts in: python/tests/export_artifacts/StatefulModel
```bash
python python/tests/manual_encoder_decoder_export.py
```

run the c++ code, with a batch size of 2.

```bash
./cpp/build/stage/bin/ExecuToolsEncoderDecoder \
--model python/tests/export_artifacts/EncoderDecoderWrapper/opus_encoder_decoder_model.pte \
--input "Hello World" "when the night has come and the land is dark."
```

This should produce the output:
```
I 00:00:00.003197 executorch:shared_memory_manager.cpp:123] Allocated normal buffer for method et_module_init, memory id 0, size 1344
I 00:00:00.003215 executorch:shared_memory_manager.cpp:135] Allocated shared buffer for method et_module_init, memory id 1, size 1344
I 00:00:00.003229 executorch:shared_memory_manager.cpp:123] Allocated normal buffer for method set_cache, memory id 0, size 2400
I 00:00:00.003232 executorch:shared_memory_manager.cpp:146] Reusing shared buffer for method set_cache, memory id 1, size 1344
I 00:00:00.003238 executorch:shared_memory_manager.cpp:123] Allocated normal buffer for method get_cache, memory id 0, size 1600
I 00:00:00.003241 executorch:shared_memory_manager.cpp:146] Reusing shared buffer for method get_cache, memory id 1, size 1344
I 00:00:00.003351 executorch:test_executorch.cpp:79] Test passed
(.executools_venv) nlong@zelfron:~/execu-tools$ ./cpp/build/stage/bin/ExecuToolsEncoderDecoder \--model python/tests/export_artifacts/EncoderDecoderWrapper/opus_encoder_decoder_model.pte \--input "Hello World" "When the night has come and the land is dark"^C
(.executools_venv) nlong@zelfron:~/execu-tools$ ./cpp/build/stage/bin/ExecuToolsEncoderDecoder \
--model python/tests/export_artifacts/EncoderDecoderWrapper/opus_encoder_decoder_model.pte \
--input "Hello World" "when the night has come and the land is dark."
I 00:00:00.000200 executorch:multi_entry_point_runner.cpp:22] MultiEntryPointRunner: Initializing executorch program
I 00:00:00.032356 executorch:multi_entry_point_runner.cpp:26] MultiEntryPointRunner: Initializing shared memory manager
I 00:00:00.037710 executorch:shared_memory_manager.cpp:123] Allocated normal buffer for method et_module_init, memory id 0, size 8496752
I 00:00:00.041594 executorch:shared_memory_manager.cpp:135] Allocated shared buffer for method et_module_init, memory id 1, size 6170224
I 00:00:00.158923 executorch:shared_memory_manager.cpp:123] Allocated normal buffer for method reset_encode_prefill, memory id 0, size 189229328
I 00:00:00.158938 executorch:shared_memory_manager.cpp:146] Reusing shared buffer for method reset_encode_prefill, memory id 1, size 6170224
I 00:00:00.274785 executorch:shared_memory_manager.cpp:123] Allocated normal buffer for method decode, memory id 0, size 186694416
I 00:00:00.274801 executorch:shared_memory_manager.cpp:146] Reusing shared buffer for method decode, memory id 1, size 6170224
I 00:00:00.274856 executorch:multi_entry_point_runner.cpp:32] MultiEntryPointRunner: Loading methods
I 00:00:00.278592 executorch:multi_entry_point_runner.cpp:37] MultiEntryPointRunner: Constructor Complete
I 00:00:00.278599 executorch:encoder_decoder_runner.cpp:29] EncoderDecoderRunner: Initializing event tracer
I 00:00:00.278603 executorch:encoder_decoder_runner.cpp:31] EncoderDecoderRunner: Setting event tracer debug level
I 00:00:00.278606 executorch:encoder_decoder_runner.cpp:34] EncoderDecoderRunner: Setting event tracer profiling level
I 00:00:00.278610 executorch:encoder_decoder_runner.cpp:41] EncoderDecoderRunner: Initializing tokenizer
I 00:00:00.279587 executorch:string_tokenizer.cpp:9] HFStringTokenizer: Loading HF Tokenizer blob, beginning with: {"version":"1.0","tr ...
I 00:00:00.346325 executorch:string_tokenizer.cpp:13] HFStringTokenizer: Successfully loaded HF Tokenizer blob, beginning with: {"version":"1.0","tr ...
Running runner, with input strings: 
  'Hello World'
  'when the night has come and the land is dark.'
I 00:00:00.352909 executorch:string_tokenizer.cpp:24] HFStringTokenizer: Encoded 2 strings, with a length of 12.
I 00:00:00.352923 executorch:encoder_decoder_runner.cpp:68] Encoded 2 strings, with a length of 12.
['<pad> Bonjour','<pad> quand',]['Monde','la',]['</s>','nuit',]['<pad>','est',]['<pad>','venue',]['<pad>','et',]['<pad>','que',]['<pad>','la',]['<pad>','terre',]['<pad>','est',]['<pad>','sombre',]['<pad>','.',]['<pad>','</s>',]
Runner ran, generated output strings:
   '<pad> Bonjour Monde</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>'
   '<pad> quand la nuit est venue et que la terre est sombre.</s>'
```

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