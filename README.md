# Tools to help standardize the export of hugging face models to be used with c++.

### this is really early, and not at all ready for use.
Split into cpp and python components.


### to run this (janky at the moment):
1. build the c++ component, this downloads a slightly modified (currently only c++ side) version of executorch repo and builds it.
2. build/install the executorch from folder as normal-> just to have the main branch executorch/ nightly torch.
3. run the python file tests/test_model_export.py -> this outputs a stupid simple single buffer with get/set functions.
4. run cpp build artifact build/stage/bin/ExecuToolsUnitTests (need to update the name...) -> this consumes the pte from step 3, and generates a shared buffer.

This currently relies on a modified branch of executorch here:
https://github.com/cptspacemanspiff/executorch/tree/personal_main


ToDo:
currently the biggest issue is that the can only be a single buffer object per memory id. currently I am relying on the greedy allocator do actually do the memory allocation, but this needs to be pulled out into a wraper separate step so that within a buffer the objects are places consistantly across methods. It also currently relies on the buffers being mutable (ie non-constants) so I have added a "self.buffer.add_(0)" to mutate them. I think this will be fixed with the memory planning stuff.

That being said, if you read this far, I would not mind feedback if others have thoughts, I originally started down this rabbit hole to get a pipeline for huggingface encoder/decoder models. but I am mainly working with full linux/android, and probably am missing some big issues as I hack this together.