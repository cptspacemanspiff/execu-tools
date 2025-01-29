## API:

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


# Overview:

> This is my understanding after just about a month of looking at this, so there is a high possibility that my understanding is incorrect. I would appreciate feedback.

When using executor to run models models with state, it can become painful. The biggest issue that I have encountered was that unlike torch.compile torch.export does not allow for data-dependent control flow.


>At least not without re-writing the model code, to explicitly use torch.cond, which in turn requires that all inputs/outputs of each functional branch are identical. 


Because of this, even models that run with torch.compile in strict mode can be hard to get working in torch.export/executorch.

## A Specific Example for Motivation: Helsinki-NLP/opus-mt-en-fr

This is a hugging face version of the OPUS translation model, which is an encoder-decoder BART style model, with a MarianMT architecture.

If one were to naively export this:

1. Export the encoder (no state, this is easy, and just works).
2. Export the decoder (has an internal state for the decode, but this can be done with executorch similar to how the llama/decoder other models handle it.)
3. glue it together on the c++ side, w/ the decoder receiving the encoder outputs, a while loop and token processing.

The issue with this is that the decoder also has to take the encoder outputs, and since there is no data-dependent branching (to check whether they have already been calculated), it will need to recalculate the cross-attention cache every time.

<details>
<summary><b>Side-Note 1:</b> <i>Why not add explicit branching torch.cond() and rewrite the model?</i></summary>

>
>Coming back to the re-writing the model, it could be possible to place a torch.cond, however the conditionals cannot have side affects. 
So:
>  1. In one branch you calculate and save the cross attention cache to the internal state buffer.
>  2. while in the other you only read the internal state buffer.
>
>However in both cases, outside the torch.cond branch the entire self attention cache always needs to be copied back to the internal state buffer, even if it was not modified. This is less than ideal, because most of the time you are only reading the self attention cache, and only modify it once.
</details>

So a solution to this is (and what this project does): 

_Just have 2 different functions that share the state._

This aligns with how many torch models are written (often a lot of state) and allows the exporter of the model to define where/how get around the limitations of torch.export.

It also keeps more of the heavy lifting in python, reducing the possibility of differences between the python implementation and the c++ runtime side.


Side Note 2: Another alternative would be to pass the kv cache state as a user input, then one can manually manage the tensor outside of the graph, and pass it between multiple graphs that modify it.
<details>

<summary><b>Side-Note 2:</b> <i>Why not pass the state as user input.</i></summary>

>The problem that comes up here is that:
>1. Thats more work on the runtime c++ side, and more chances for errors.
>2. User inputs must be tensors, one cannot pass in a arbitrary object with a bunch of registered buffers. This is a pain, for instance the static cache implementation from hugging face for the OPUS model has (5 layers x 2 key/value cache x 2 self/cross attention) = 20 registered buffers, which each have to be passed individually, and managed, in c++.
>3. Less general, say we get a new model that is nearly the same, but we have 3 layers, now the c++ side and the user side needs to change.

</details>

So shared state between multiple exported methods of a model would be useful, how do we get it working?

It can't be that hard... 

_I said a month ago...._

## First, How do we get multiple methods?

torch.export is not the most flexible, and specifically if one wants to export a module (along with it's associated state) you must export the `forward` method. More specifically the torch.export of modules relies on running `module()`, which calls the `forward` method. This is fine, I guess.. But we want to export multiple methods, potentially not all named `forward`. 

The solution to this is _don't export the model directly_, and use a wrapper class, with the model as a submodule. Then monkey-patch the wrapper classes' `forward` method with each function we want to export. 

This solves a couple problems:

1. Most models are not necessarily setup how one would like for export with minimal graph breaks (which need to be patched together on the runtime side with c++ glue code). For example in the opus model, we want to export 2 functions, a init-encode-decode_once, so that we encode the model, then run one iteration of the decoder so that we save off the cross-attention cache weights. Then when the decoder runs, the weights are already set and just need to be used.
2. We are trying to export a pseudo hugging face Transformers `generate` method (or as much as one we as we can get away with, again complex c++ adds complexity, complexity sucks to maintain), most models have a lot of logic outside the model specific, we can implement most of this in the wrapper.
3. We don't allow `forward` methods in our wrapper. This eliminates the edge case of our monkey-patched forward method itself calling forward.

This type of multiple-method export does seem to have some level of undocumented support inside the executorch export pipeline, where each method is initially run through `torch.export` independently, but then saved to a `method_name : export_program` dictionary. The later stages of to_edge and to_executorch then operate on this multi-method dictionary.

## Ok, So we have multiple methods, how do we get our shared state?

So we want to share state between multiple exported methods, how do we do that?

<details>

<summary><b>Side-Note 3:</b> <i>How memory and storage works in the executorch export process and runtime.</i></summary>

>Memory in executorch is all pre-planned. During the export of a model function, in the final steps of export, the graph is walked and each op-argument is associated with a memory location, consisting of a mem_id, an offset into that id, and a size. It will keep track of lifetimes, and reuse locations in memory if possible (if you use the greedy algorithm). 
>
>On the runtime c++ side, we are told the max size of each mem_id, and create a buffer for each one during initialization. Then when executing model method, the pointer arithmetic is calculated, each op is ran, and the operation is given the pointer location within our created buffers for its inputs and outputs.
>
>One thing to note is the concept of memory IDs / memory arenas. The purpose of this is so that the runtime can do different allocations for different types of storage, particularly regarding the backends. IE GPU memory vs Main memory, or SRAM/DRAM on some weird FPGA device. 
>
>In any case, each memory ID is allocated separately, as chosen by the runtime implementer, and currently most things seem to use a single memory ID due to many phones/edge devices having a single type of memory.
</details>

So, with the above in mind, if we could somehow get the pointers to the internal state buffers to point to the same location in memory at runtime, we would have shared state between all the methods of our exported model.

### How to have shared memory pointers at runtime:

The first thing that needs to be done, is during export, we need to make sure that all shared buffer locations are identical between methods. The easiest way to do this is to place them into a unique shared mem_id. Then on the runtime side, when we instantiate that mem_id, we reuse the memory block between multiple methods.

This works, and is fairly straightforward, mainly due to the multiple memory arenas, and the fact that the memory planners are designed to account for different backend providers having some objects in different IDs. So assuming that we have a list of shared buffers, in our custom memory planning pass we check if a buffer object is `shared` and if so, set it's mem_id to a custom location the special unique value: `2`. 

Of course this runs into the issue of how do we ensure that the memory layout within a memory id is consistent between methods?

There are two issues:
1. The order of objects in a the mem_id is dependent on the order in which they are encountered in the export graph.
2. Not all methods access all buffers.

To solve this, we create a synthetic method, which is automatically generated from our list of shared buffers. All this method does is go through each buffer and modifies them (currently sets them to 0). We call hard code this method to be called `et_module_init`.

Now when we go to generate the memory plan we have a couple steps:
1. we run the generation process on our synthetic `et_module_init` method first, that generates a layout that includes every shared buffer. 
2. Save off this memory plan for our shared mem_id.
3. Rerun memory planning on all methods, placing any shared objects used into mem_id: `2`
4. Once the plan is done, overwrite the plan for each object in mem_id 2 with the plan that we previously generated.

And viola, we have a common set of memory locations for all shared buffers across all methods, we even got an initialization method for free.

However, there is an issue, namely the case where a shared buffer is used but not modified.

### What if our method does not modify the buffer?

<details>
<summary> <b>Side-Note 4:</b> <i>Export handling of buffers, constants, parameters and inputs</i></summary>
 
>The torch.export/executorch pipeline handles buffers, constants, parameters and inputs differently. During torch.export and to_edge, the graph is functionalized, and any side effects, or values used in the graph are lifted to the graph inputs and outputs.
>
>For Instance:
>
>What was initially:
>```python
>inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
>```
>After torch.export:
>```
>%mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%embedding, 22.627416997969522), kwargs = {})
>```
>After to_edge:
>```
>%aten_mul_tensor : [num_users=1] = call_function[target=torch.ops.aten.mul.out](args = (%aten_embedding_default, %_lifted_tensor_constant29), kwargs = {out: %alloc_2})
>```
>
> At the same time, there is a separate graph signature object. This keeps track of input/output and their behavior. Particularly it will keep track 
>
>
</details>

This causes issues because the export pipeline treats constant buffers and mutable buffers differently.
1. **constant buffers:** what is inferred if we access a registered buffer of a pytorch model, but do not modify it.
2. **mutable buffers,** what is inferred if we modify a registered buffer.

During the to_edge part of the export process, the method graph of each exported method is analyzed and the registered buffer inputs are classified as constants, or mutations. 

If they are determined to be constant, their values will be pulled from some memory location inside the pte, or an externally provided file. In either case this is not what we want, because while a particular method does not modify the buffer, other methods might.

Handling this properly would involve digging around in the to_edge export process, possibly adding an additional buffer type, and having a generally painful day.

So we can do the stupid thing, and make sure that in all of our exported methods we modify the buffer.

Specifically, after exporting a model that uses a shared buffer `cache` we add a copy operation at the start of the graph that is equivalent to:
```python
def get_cache(self, x):
    self.cache.copy_(self.cache) # we auto-add this operation in.
    x.copy_(self.cache)
    return None
```
This is obviously idiotic, and pointless, since the value of cache has clearly not changed. However for our purposes, to_edge looks at this and sees that cache is a mutable buffer, which is marked appropriately in the graph signature. Additionally because figuring out if a buffer gets mutated is hard, we do this for all our methods, even methods like:

```python
def set_cache(self, x):
    self.cache.copy_(x) # auto added in, even though we were already mutating self.cache
    self.cache.copy_(x)
```

After to_edge, we go through the graph and remove any copy operation where both the source and target are the same location, which also in our shared buffer list. 

Finally, For `get_cache()`, fact that we have mutated the cache value is no longer in the graph. However it does remain in the graph signature, and in the final stage of export copies to buffers are added back in, as long as they are not pointing to themselves.

###

>PS: If you read this far, I am just finishing off taking for a year long travel trip, and am currently unemployed and am actively looking for work.
>
>I started down this rabbit hole b/c I wanted to do on-device AI for a side-project, to get more experience in the types of projects I want to work on and to have something to show off. 
>Also building this is way more fun and interesting than throwing my resume into the ether when applying for jobs and grinding leetcode. 
>
>If you found it useful, are hiring (or know someone), I would love to work on this type of thing and get paid for it.
>
> My objective is to build cool AI related stuff with awesome people, I currently live on the east coast, but am willing to relocate to most anywhere... but not Texas.
>
>[LinkedIn](https://www.linkedin.com/in/nicholas-long-z42/)

```