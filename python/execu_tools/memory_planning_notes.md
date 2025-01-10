# Memory Planniing Notes:

The memory planner in executorch is a pain in the rear...

But anyway, these are notes I made for understanding the process and the anticipated outputs and side-effects of each of the stages.

interesting side-notes:
https://dev-discuss.pytorch.org/t/a-new-strategy-for-automatic-custom-operators-functionalization/2733

# EdgeProgramManager.to_executorch in _program:

* weights_to_outputs_pass: checks what nodes are gradiants, and does magic to output them -> ignored if no gradients.
* unsafe_remove_auto_functionalized_pass: previosly in torch export, higher order ops 
* insert_write_back -> modifies the program graph in place inserting a return copy for mutable buffer at end of graph. Since     torch.ops.aten.copy_.default, is in the to_out_var_skiplist, this will not be converted to out variant in ToOutVar step. All other copy ops are exir.copy, so will have memor backing.
* Goes through additional passes:
  * user-specified pass list from ExecuTorchBackendConfig
  * SpecPropPass() -> adds tensor specs to nodes.
  * edgeToBackendOpsPass -> converts to backend?, 1-1 replaceing ops?
  * remove asserts

  * view aliasing
    * normalize view copy-> sets all views to be based off of non-view tensor, 
    * removes dead code (ie nodes that new views do not depnd on.)
    * replace the view copy w/ view
  * SymShapeEvalPass -> converts dynamic shapes to upper bounds.
  * ToOutVarPass -> converts to functional with allocs in between.

  * run memory planning pass -> updates the graph

  * run external constants pass -> updates the graph 

# what we need to do to have consistant memory layout for all program methods:

1. place the memory in its own region before calling the planner.
2. update the lifetime, so that the planner does not delete.
3. after the planner runs overwrite the planned memory for our memory ID. (we 'own' the whole ID, so this is fine)
  1. we use a precomputed memory layout that is shared between methods, and for each shared buffer in the method, we look up the appropriate offset for the ID.

issues: if a buffer is shared, but not mutated, it is not exported after memory planning correctly (treated as a const buffer), we need to change the signature to treat it as a mutable buffer, even though it is not. However there are a couple things that touch the 'mutable buffer'

### Where is buffers_to_mutate used?

* _is_mutable_buffer in exir/memory_planning.py
  for determining if input/output need to be allocated, mutable buffer is skipped, so is const buffers.
* _is_mutable_buffer in exir/passes/spec_prop_pass.py
  * updates the spec.const parameter in all placeholder node specs.
* _find_fqn_for_placeholder in exir/emit/_emitter.py
* tag_constant_data in exir/backend/utils.py -> used in a bunch of backends. 
* tag_mutated_buffer in exir/backend/utils.py -> only used in coreml partitioner for pulling out mutable state. not sure how this affects stuff.

issues: alignement requirements of the buffers? Are there ops that require specific alignment that we may not know about ahead of time? when we do the planning probably need to specify an alignment.


