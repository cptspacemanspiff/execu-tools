# Overview (And my rational for working on this):

I started this project as I was trying to get models to run on edge devices, and this was my attempt at making the general case easier.

Currently there seem (to my neophyte eye) to be 3 avenues for getting models onto a device:

* TensorFlow-lite: 
  * Older, less support for newer hardware and models archetectures (statefull models)
  * Tied into tensorflow ecosystem and maintained by google.
* GGML: 
    * No dependecies, all c++, no funky compiler process.
    * Good community support.
    * Good if your model post-processing is supported, less so if you need to go off the beaten path.
* Executorch: 
  * Evolution of pytorch on device pipeline, seems to have good buy-in from hardware manufacturers.
  * Relies on torch.compile infrustructure so it gets attention from those working on the server side.
  * Uses pytorch, so most models, if written in a torch compileable manner can be exported with minimal fuss.
  * Has pretty good validation/ on device testing tooling.
  * New, and has some rough edges, statefull models have pain points.
* Onnx:
  * (TODO: pros/cons onnx)

Out of these I decided that in the longer term executorch seems to be the most maintainable and has the most resources. Also personally want to have a better understanding of pytorch so decided to start there.

# A specific use case:

So for a specific motivating usecase, I want to run a encoder-decoder model on my edge device. Specifcally A Marian (BART Architechure) machine translation model `Helsinki-NLP/opus-mt-en-fr`. Even more specifically I want to use the implementation from the Hugging Face Tranformers library.

## Current state and why it fails:

While there are a few issues on the Hugging Face side of this, that is mostly getting sorted out as they go through and make their models torch compile compatable, and are relatively straightforward chenges to make.

The bigger issues come with using executorch to export the model to device.

First this is a generation process with state, and ideally one would only have to hand the generator function 3 things:
  * The encoder tokens.
  * The encoder attention mask.
  * A decoder prompt that is used in a prefill stage.

In a perfect world:
1. the encoder runs.
2. the decoder runs in a prefill mode:
  * Calculating the cross-attention cache for the entire encoder sequence
  * Updating the self attention cache for the prefill tokens.
  * Decode the next logits from the prefill.
  * Process the logits.
  * Determine the next-token (greedy, beamsearch, etc)
3. Run the decoder in regular mode:
  * Take the next-token from the previous step use it to decode the logits for the next token.
  * Process the logits.
  * Determine the next-token (greedy, beamsearch, etc)
  * Repeat until the stopping condition is met.

This is how (roughly) the generate function in HF Transformers or else where works. But it has a few issues when trying to export with executorch.

The biggest issue is Data-dependent control flow. Torch export does not support data-dependent control flow (without explicitly adding torch._check and torch.cond), and there is alot of it in this process. This is a major difference between torch.compile and torch.export.

In the workflow above there is at least the following:
1. whether the encoder runs and the the decoder is in prefill mode, vs just running the decoder is data-dependent. 
2. Greedy search is not data-dependent, but beamsearch is as it rearanges k-v cache based on the data.  
3. The The while loop and stopping condition is data dependent.   

The way this is handled for llama text models is to lift all these data-dependent branching out of the model, and implement it in c++. This works for decoder models because there is not cross-attention/encoder that only runs once, and you can pass in a cache-position value and then mask off future positions in the model, so you do not calculate more than needed.

When shifting this methodology to encoder-decoder models, this causes issues, because you really want to calculate the cross attention cache once, and while you could pass a cross attention cache into the model as inputs, this involves rewriting the encoder-decoder model's python implementation.

Another issue with this methodology is that the logit processing is done outside of the model (so it needs to be rewritten from a reference python implememtation into c++), and there is no way to modify the kv cache without going having the decoder calculate those values. While works for greedy/sampler decoder strategies, more complex search such as beam search does not support this.

Anyhow there ends up being portions of the model implementation split between different parts of the code, which makes everything harder to maintain/test/validate and adapt new models for on device.

## Possible solutions:

### Explictly add control flow with torch.cond:

One possibility is to have a single exported function that has control flow hacked into it. For example, once can have a single model that has an additional input that is `run_the_encoder` then internally you explicitly user torch.cond to branch on that condition.

The issue with this is in how torch export and executorch handles internal state. Specifically the exported programs are functionalized, and any internal state reading is lifted into the signature as input, while any buffer mutations are added as outputs. Then in the final export these inputs/outputs are automatically in, and there is a write-back pass that copies any mutations to the input data localtions.

This means that even if the encoder is only ever read, there will still be a copy op on the output (maybe? idk... probably thinking too much...)... wait this might actually work... because the writeback step only occurs if the buffer mutation actually happens, if on the other hand it gets optimized out, then it sees it's copying from the input to the output, and aside from forcing the module to have additional inputs (common state) that may be unused we do not have to do anything. 

It is probably simpler than shared state.

### Allow for Shared State between programs.

In this case we spearate the overall processing into different steps, and can run them independently, where any common state (kv-cache) is shared.

Functionally at the output this is the same as lifting the kv cache out and providing it as an input, but ergonomically it does not require rewriting the python code to support this, and itmakes the c++ side much simpler to deal with. 

For example on the c++ side it becomes:

1. init program.
2. run encoder module
3. run decoder prefill module.
4. run decoder:
  1. run decoder -> logits
  2. process logits -> tokens (greedy sampling etc)
  3. process stopping criteria -> return finished flag

The C++ code simply needs to call these in order, and then stop when the finished flag is true. 

This allows for bettwe maintainability by shifting all the logic to the python side.




