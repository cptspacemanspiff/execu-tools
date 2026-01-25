import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from torch.export import Dim, export

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, 3),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1,1))
        )
        self.linear = torch.nn.Linear(16, 10)

    def forward(self, x):
        y = self.seq(x)
        y = torch.flatten(y, 1)
        y = self.linear(y)
        return y

model = Model()
inputs = (torch.randn(1,1,16,16),)
dynamic_shapes = {
    "x": {
        2: Dim("h", min=16, max=1024),
        3: Dim("w", min=16, max=1024),
    }
}

exported_program = export(model, inputs, dynamic_shapes=dynamic_shapes)
executorch_program = to_edge_transform_and_lower(
    exported_program,
    partitioner = [XnnpackPartitioner()]
).to_executorch()

with open("model.pte", "wb") as file:
    file.write(executorch_program.buffer)