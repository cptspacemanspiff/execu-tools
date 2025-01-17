import torch
from torch.export import ExportedProgram
from executorch.exir import to_edge, to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
class TestBufferMutationBug(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("buffer", torch.ones(1))

    def forward(self, x):
        # set_cache type op.
        # self.buffer = x
        y = x + 0
        self.buffer.copy_(y)


        # this is a hack around the bug, last op cannot be a mutable _copy op.
        # Specifically to_edge in executorch uses run_decompositions to convert to aten operators.
        # internally this does some stuff to the graph to get the aten operators,
        # and as a last step of _decompose_and_get_gm_with_new_signature_constants, 
        # checks if the output is being fed by a mutable _copy and removes it from the graph.
        # 
        # This is in torch.export.exported_program.py, _remove_unneccessary_copy_op_pass
        # and was intrduced in : pull request #134801
        # as a hack to reproduce show that if the last op is not _copy the there are no issues.:
        # self.buffer.add_(0)
        return None


# Create and run the module once
module = TestBufferMutationBug()
module(torch.tensor(2))

print("After running module once, buffer value is:", module.buffer)

# Export the module to TorchScript
print("\nExported program graph (shows original operations):")
exported_program : ExportedProgram = torch.export.export(module, (torch.tensor(2),))
exported_program.graph.print_tabular()

# Convert to edge IR format
print("\nEdge IR graph (after decomposing to ATen operators):")
edge_program = to_edge(exported_program)
edge_program._edge_programs['forward'].graph.print_tabular()

executorch_program = edge_program.to_executorch()

print("done")

# # Convert to XNNPACK format (occurs here as well)
# print("\nXNNPACK graph (after partitioning for XNNPACK backend):")
# xnnpack_program = to_edge_transform_and_lower(exported_program, partitioner=[XnnpackPartitioner()])
# xnnpack_program._edge_programs['forward'].graph.print_tabular()
