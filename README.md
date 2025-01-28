# Overview:

(This is my understanding after just about a month of looking at this, so there is a valid possibility that my understanding is incrorrect. I would appreciate feedback.)

When using executorch to run models models with state, it can become painful. The biggest issue that I have encountered was that unlike torch.compile torch.export does not allow for data-dependent control flow.

```
at least not without re-writing the model code, to explicitly use torch.cond, which in turn requires that all inputs/outputs of the functional branch are identical). 
```

Because of this, even models that run with torch.compile in strict mode can be hard to get working in torch.export/executorch.

## A Specific Example for Motivation: Helsinki-NLP/opus-mt-en-fr

This is a hugging face version of the OPUS translation model, which is an encoder-decoder BART style model, with a MarianMT archetecture.

If one were to niavely export this:

1. Export the encoder (no state, this is easy, and just works).
2. Export the decoder (has an internal state for the decode, but this can be done with executorch similar to how the llama/decoder other models handle it.)
3. glue it together on the c++ side, w/ the decoder recieving the encoder outputs, a while loop and token processing.

The issue with this is that the decoder also has to take the encoder outputs, and since there is no data-dependent branching (to check whether they have already been calculated), it will need to recalculate the cross-attention cache every time.

```
Side Note: coming back to the re-writing the model, it could be possible to place a torch.cond, however the conditionals cannot have side affects. 
So:
  1. In one branch you calculate and save the cross attention cache to the internal state buffer.
  2. while in the other you only read the internal state buffer.
However in both cases, outside the torch.cond branch the entire self attention cache always needs to be copied back to the internal state buffer, even if it was not modified. This is less than ideal, because most of the time you are only reading the self attention cache, and only modify it once.
```

So a solution to this is `just have 2 different functions that share the state`. This aligns with how many torch models are written (often a lot of state) and allows the exporter of the model to define where/how get around the limitations of the export.

It also keeps more of the heavy lifting in python, reducing the possibility of differences between the python implementation thats exported and the c++ runtime side.

```
Another alternative would be to pass the kv cache state as a user input, then one can manually manage the tensor outside of the graph, and pass it between multiple graphs that modify it.

The problem that comes up here is that:
1. Thats more work on the runtime c++ side, and more chances for errors.
2. User inputs must be tensors, one cannot pass in a arbitrary object with a bunch of registered buffers. This is a pain, for instance the static cache implementation from hugging face for the OPUS model has (5 layers x 2 key/value cache x 2 self/cross attention) = 20 registered buffers, which each have to be passed individually, and managed, in c++.
3. Less general, say we get a new model that is nearly the same, but we have 3 layers, now the c++ side and the user side needs to change.
```

So shared state between multiple exported methods of a model would be usefull, how do we get it working?

It can't be that hard... _I said a month ago...._

## First, How do we get multiple methods?

torch.export is not the most flexible, and specifically if one wants to export a module (along with it's associated state) you must export the forward method. More specifically torch.export of modules relies on running module(), which calls the forward method. This is fine, I guess.. But we want to export multiple methods, potentially not all named `forward`. 

The solution to this is `don't export your module directly`, use a wrapper class, with our model as a submodule and monkey-patch the wrapper classes forward method. 

This solves a couple problems:

1. Most models are not neccisarilly setup how one would like for export with minimal graph breaks (which need to be patched together on the runtime side with c++ glue code). For example in the opus model, we want to export 2 functions, a init-encode-decode_once, so that we encode the model, then run one iteration of the decoder so that we save off the cross-attention cache weights. Then when the decoder runs, the weights are allready set and just need to be used.
2. We are trying to export a psuedo hugging face Transformers `generate` method (or as much as one we as we can get away with, again complex c++ adds complexity, complexity sucks to maintain), most models have alot of logic outside the model specific, we can implement most of this in the wrapper.
3. We don't allow `forward` methods in our wrapper. This eliminates the edge case of our monkey-patched forward method itself calling forward.

This type of multiple method export does seem to have some level of undocumented support inside the executorch export pipeline, where each method is initially run through `torch.export` independently, but then saved to a `method_name : export_program` dictionary. The later stages of to_edge and to_executorch can operate on this multi-method dictionary.

## Ok, So we have multiple methods, how do we get our shared state?

So we want to share state between multiple exported methods, how do we do that?

```
Sidetrack: How memory and storage works in the executorch export process and runtime:

Memory in executorch is all pre-planned. During the export a model function, during the final steps of export, the graph is walked and each op-argument is associated with a memory location, consisting of a mem_id, an offset into that id, and a size. It will keep track of lifetimes, and reuse locations in memory if possible (if you use the greedy algorithm). 

On the runtime c++ side, we are told the max size of each mem_id, and create a buffer for each one during initialization. Then when executing model method, the pointer arithmatic is calculated and when each op is ran, the operation is given the pointer location within our created buffers for it's inputs and outputs.

One thing to note is the memory IDs / memory arenas, the purpose of this is so that the runtame can do different allocations for different types of storage, particularly regarding the backends. IE GPU memory vs Main memory, or SRAM/DRAM on some wierd FPGA thing. 

In anycase each memory ID is allocated separately, as chosen by the implementer, and currently most things seem to use a single memory ID due to many phones/edge devices having a single type of memory.
```

With the above in mind, if we could somehow get the pointers to the internal state buffers to point to the same location in memory at runtime, we would have shared state between all the methods of our function.

### How to have shared memory pointers at runtime:

The first thing that needs to be done, is during export, we need to make sure that all shared buffers locations are the same between methods. The easiest way to do this is to place them into a shared mem_id, then on the runtime side, when we instatiate that mem_id, we reuse the buffer between the multiple methods.

This works, and was very easy to initially get working, but has an issue of how do we gurantee that the layout of objects within a memory ID is the same between all methods.