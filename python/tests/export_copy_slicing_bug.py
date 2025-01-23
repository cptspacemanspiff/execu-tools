import torch
from torch.export import ExportedProgram, Dim
from executorch.exir import to_edge, to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

from executorch.runtime import Runtime, Program


class TestSlicingMutationIssue(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("cache", torch.zeros(4, 4))
        
    # x is a 3x3 or 4x4 matrix.
    def forward(self, x):
        self.cache[0:x.shape[0], 0:x.shape[1]] = x
        return self.cache

x_in_out_12 = torch.ones(1, 2)
x_in_out_22 = torch.ones(2, 2)
x_in_out_33 = torch.ones(3, 3)
x_in_out_44 = torch.ones(4, 4)

# Create and run the module once eager mode.
module = TestSlicingMutationIssue()
# module(x_in_out_33)
# print(f"After running module once, buffer value is:\n{module.cache}")
dim_0_max = 10
dim_1_max = 20
# dims that work:
# dim_0 = Dim("dim_0", min=1, max=4)
# dim_1 = Dim("dim_1", min=1, max=4)

# dims that don't work:
d_0 = Dim("dim_0", min=1, max=dim_0_max)
d_1 = Dim("dim_1", min=1, max=dim_1_max)

exported_program = torch.export.export(
    module, (x_in_out_33,), dynamic_shapes={"x": {0: d_0, 1: d_1}}
)
print('--------------------------------')
print(exported_program.graph)
print('--------------------------------')

# quit()
edge_program = to_edge(exported_program)
print('--------------------------------')
print(edge_program._edge_programs["forward"].graph)
print('--------------------------------')
executorch_program = edge_program.to_executorch()
executorch_program.save("export_copy_slicing_bug.pte")
print('--------------------------------')
print(executorch_program._execution_programs['forward'].graph)
# run the program:

et_runtime: Runtime = Runtime.get()
program: Program = et_runtime.load_program(
    "export_copy_slicing_bug.pte",
)
print("Program methods:", program.method_names)

print("Running forward method")
forward_method = program.load_method("forward")

# fills out the ones subarray in the cache
out = forward_method.execute([x_in_out_22])
print(f"After running module once, buffer value is:\n{out}")

# should have no effect (value is unchanged from previous cache)
out = forward_method.execute([x_in_out_12])
print(f"After running module once, buffer value is:\n{out}")

# should increase the size of the ones subarray in the cache
out = forward_method.execute([x_in_out_33])
print(f"After running module once, buffer value is:\n{out}")

# should work, but does not because of the dynamic shape bug.
# out = forward_method.execute([x_in_out_44])
# print(f"After running module once, buffer value is:\n{out}")
