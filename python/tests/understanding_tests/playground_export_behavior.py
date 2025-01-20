import torch
from torch.export import export, ExportedProgram
from torch.export.dynamic_shapes import Dim

from executorch.exir import to_edge

class BranchingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer("cache", torch.tensor(-1))

    # this does not work, b/c branching on shape:
    # https://pytorch.org/docs/stable/export.html#data-shape-dependent-control-flow
    # def forward(self, x):
    #     output = 0
    #     if x.shape[0] == 0:
    #         output = 0
    #     if x.shape[0] == 1:
    #         output = 1
    #     if x.shape[0] == 2:
    #         output = 2
    #     else:
    #         output = 3
    #     return torch.tensor(output)

    # def forward(self, x):

    #     def check_0(x: torch.Tensor):
    #         return torch.tensor(0)

    #     def check_1(x: torch.Tensor):
    #         return torch.tensor(1)

    #     def check_2(x: torch.Tensor):
    #         return torch.tensor(2)

    #     def check_3(x: torch.Tensor):
    #         return torch.tensor(3)

    #     # # output = torch.tensor(0)
    #     output = cond(x.shape[0] == 1, check_1, check_0, (x,))
    #     output = cond(x.shape[0] == 2, check_2, check_0, (x,))
    #     output = cond(x.shape[0] == 3, check_3, check_0, (x,))
    #     return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.tensor(0)
        def check_0(x: torch.Tensor):
            return x # seems to need retuning an input, cannot just create a new tensor and return it...

        def check_1(x: torch.Tensor):
            self.cache.fill_(1)
            return x.fill_(1)

        def check_2(x: torch.Tensor):
            self.cache.fill_(2)
            return x.fill_(2)

        def check_3(x: torch.Tensor):
            self.cache.fill_(3)
            return x.fill_(3)

        output = torch.cond(x.shape[0] == 1, check_1, check_0, (output,))
        output = torch.cond(x.shape[0] == 2, check_2, check_0, (output,))
        output = torch.cond(x.shape[0] == 3, check_3, check_0, (output,))
        return self.cache, output


def test_branching_export():
    # Create model and exporter
    model = BranchingModel()
    original_output = model(torch.ones(3))
    # Register the forward method with example input
    example_input = torch.ones(3)
    # Export the model
    method_graph: ExportedProgram = export(
        model,
        (example_input,),
        kwargs=None,
        dynamic_shapes={"x": {0: Dim("batch_size", min=1, max=10)}},
        strict=False,
    )

    print(method_graph.graph.print_tabular())
    # Test different batch sizes
    test_cases = [
        torch.ones(0),  # empty batch
        torch.ones(1),  # batch size 1
        torch.ones(2),  # batch size 2
        torch.ones(3),  # batch size 3
    ]

    # Verify the exported model produces the same outputs
    for input_tensor in test_cases:
        # Run original model
        eager_prev_output, eager_output = model(input_tensor)
        # Run exported model
        exported_prev_output, exported_output = method_graph.module()(input_tensor)

        assert torch.equal(eager_prev_output, exported_prev_output), (
            f"Mismatch for input shape {input_tensor.shape}: "
            f"eager={eager_prev_output}, exported={exported_prev_output}"
        )
        assert torch.equal(eager_output, exported_output), (
            f"Mismatch for input shape {input_tensor.shape}: "
            f"eager={eager_output}, exported={exported_output}"
        )

    # export to edge -> does not work b/c torch.cond canoot return aliased inputs
    # edge_model = to_edge(method_graph)
    # print(edge_model)

if __name__ == "__main__":
    test_branching_export()

