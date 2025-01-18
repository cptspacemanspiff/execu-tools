# What is this library?

My high level rational for this libraries existance is to make the export process to an executorch model, simple and easy. The Executorch people have done a really good job with the backend, but the API is pretty clunky, and exporting models with state is a pain.

This is exacerbated by the fact that most often one does not want to rewrite a model from scratch just to export the thing.

Additionally, to use a exported artifact reliably and maintainably, one needs to do more than just export a single pass through a decoder.

Ideally there needs to be a standardized interface on both the model export side and the C++ calling side, This ideally includes initialization info, automatic test running, etc. to make sure that the exported model works properly on the end device hardware (We don't want to deploy a model that ends up encountering a unknown issue on a particular backend/cpu/archetechure).

On top of this, there is alot of surrounding code that needs to be implemented to support the model.

Particularly, in decoder models:
* logit processing steps.
* search algorirhm (greedy, beam-search, etc.),
* stopping conditions (max length, eos token)
* tokenizer (probably not possible, yet)

Each one of these is model specific, can be tricky to get right, can be a pain to program, and are often written in python... It sure would be nice to export them, and just have them embedded as part of the model, and then have the c++ code call them in a standardized way.

Ideally one would be able to take a arbitrary hugging face model, and in a few lines of code export it, and then be able to run it on device, without having to reimplement the model on the C++ side.

## A Specific Motivating Example.

#### T5 Text translation:

This is a encoder-decoder model.

For a given generation, the steps are:
1. Run the encoder(encoder_inputs, encoder_attn_mask) -> returns the encoder output embedding.
  * The encoder has a dynamic sequence length (up to some maximum)
  * The encoder has a dynamic batch size (up to some maximum)
2. Run the decoder, pass in the encoders output.
  * On the first step of the process the decoder needs to do a `prefill` step, where multiple cache positions are added.
  * The encoder output is passed in, and on the first decoder step the cross-attention cache is calculated, this result is then used for the remainder of the processing.
  * T