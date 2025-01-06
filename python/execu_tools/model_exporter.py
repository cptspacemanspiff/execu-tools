# file that helps generate the exporter
from enum import Enum
from pathlib import Path
from typing import Callable, Optional
import torch
from torch.export import (
    export,
    export_for_training,
    ExportedProgram,
    ExportGraphSignature,
)
from executorch.exir import to_edge, to_edge_transform_and_lower, EdgeProgramManager
from executorch.exir import ExecutorchProgram, ExecutorchBackendConfig
from executorch.exir.pass_base import PassResult
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass, greedy
from executorch.exir.memory_planning import materialize_buffer, _is_mutable_buffer

from contextlib import contextmanager

import inspect

from executorch.exir.dynamic_shape import DynamicMemoryPlanningMode

from executorch.exir.tensor import ALIGNMENT


@contextmanager
def patch_forward(obj: torch.nn.Module, new_method):
    """
    Patches the forward method of a PyTorch module, needs to patch the class not
    just the instance b/c otherwise export will error out when trying to produce
    items that reference self.
    """
    original_method = obj.__class__.forward
    obj.__class__.forward = new_method
    try:
        yield
    finally:
        # Restore the original method
        obj.__class__.forward = original_method

def _is_buffer(
    node: torch.fx.Node, graph_signature: Optional[ExportGraphSignature] = None
) -> bool:
    """
    Check if the node is a buffer according to the provided graph signature.
    """
    if node.op == "placeholder":
        if isinstance(node.target, str):
            if node.target in graph_signature.inputs_to_buffers:
                return True
    return False

class SharedMemoryPlanningPass(MemoryPlanningPass):
    def __init__(
        self,
        shared_buffers: set[str] = set(),
        **kwargs,
    ):
        self.shared_buffers = shared_buffers
        super().__init__(**kwargs)

    def run(
        self,
        graph_module: torch.fx.GraphModule,
        graph_signature: Optional[ExportGraphSignature],
    ) -> PassResult:
        for subgm in graph_module.modules():
            if not isinstance(subgm, torch.fx.GraphModule):
                continue
            for node in subgm.graph.nodes:
                if _is_buffer(node, graph_signature):
                    buffer_name = graph_signature.inputs_to_buffers[node.target]
                    if buffer_name in self.shared_buffers:
                        # shared mutable buffers are always mem_id 2 -> there is a issue regarding layout. currently can only have a single shared mutable buffer.
                        node.meta["spec"].mem_id = 2

                        # this is a shared mutable buffer, its lifetime is infinite (max int64):
                        # this should not be updated by the memory planner, since lifetime can only expand.
                        node.meta["spec"].lifetime = [0, 9223372036854775807]
                        # this is a shared mutable buffer, even if we do not modify it, treat it as non-const.
                        node.meta["spec"].const = False 

        parent_result = super().run(graph_module, graph_signature)

        # TODO: gathar diagnostic info/ validate consistancy across all methods
        print("done")
        return parent_result


# exporter class that wraps the module, and provides methods for exporting.
# 1. Init with a module.
# 2. Register methods of the module that you want to export (including dynamic Dims).
# 3. TODO: quantize the model.
# 4. trace the model (export)
# 5. to_edge the modeln (optionally with backend).
# 6. save the executorch model.
class Exporter:
    model: torch.nn.Module
    registered_method_dict: dict[str, tuple[callable, dict]]
    registered_shared_buffers: set[str]

    method_graphs: dict[str, ExportedProgram]

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.registered_method_dict = {}
        self.registered_shared_buffers = set()
        # intermediary states:
        # self.quantized_model  :dict[str, ExportedProgram] = None #from quantize
        self.method_graphs: dict[str, ExportedProgram] = {}  # from export
        self.edge_program: EdgeProgramManager = None  # from to_edge
        self.executorch_program: ExecutorchProgram = None  # from to_executorch

    def register(self, fn, **kwargs):
        # validate that the function is a method of the module.
        if fn.__qualname__.split(".")[0] != self.model.__class__.__name__:
            raise ValueError(
                f"Function {fn.__name__} is not a method of {self.model.__class__.__name__}"
            )
        self.registered_method_dict[fn.__name__] = (fn, kwargs)

    def register_shared_buffer(self, buffer_name: str):
        self.registered_shared_buffers.add(buffer_name)

    def export(self) -> dict[str, ExportedProgram]:
        with torch.no_grad():
            for method in self.registered_method_dict:
                fn, kwargs = self.registered_method_dict[method]
                # update the forward method of the model:
                with patch_forward(self.model, fn):
                    sig = inspect.signature(self.model.forward)
                    param_names = [param.name for param in sig.parameters.values()]
                    print(param_names)
                    example_args = {}
                    dynamic_shapes = {}
                    for param in param_names:
                        if param not in kwargs:
                            raise ValueError(
                                f"Parameter {param} not found in function {fn.__name__} registration, options are {kwargs.keys()}"
                            )
                        else:
                            param_value = kwargs[param]
                            # parameter has been registered:
                            if type(param_value) is not tuple:
                                example_args[param] = param_value

                            else:
                                # we got a tuple
                                example_args[param] = param_value[0]
                                if len(param_value) > 1:
                                    # validate that the dynamic shapes are valid:
                                    # create a list of valid dims:
                                    valid_dims = range(0, len(param_value[0].shape))
                                    for dim_idx in param_value[1]:
                                        if dim_idx not in valid_dims:
                                            raise ValueError(
                                                f"Invalid dim index {{{dim_idx}}} for parameter {{{param}}} of fn {{{fn.__name__}}}"
                                            )

                                    dynamic_shapes[param] = param_value[1]
                    print(
                        f"Exporting {fn.__name__} with args {example_args}, and dynamic shapes {dynamic_shapes}"
                    )
                    method_graph: ExportedProgram = export(
                        self.model,
                        (),
                        kwargs=example_args,
                        dynamic_shapes=dynamic_shapes,
                        # strict=True
                    )
                    self.method_graphs[fn.__name__] = method_graph

        return self.method_graphs

    # def quantize_model(self):
    #     # check that we have a model traced (model graphs not empty):
    #     if len(self.method_graphs) == 0:
    #         raise ValueError("No method graphs found. Please trace the model first.")
    #     pass

    def to_edge(self, partitioners: list = None) -> EdgeProgramManager:
        # export the model graphs to the edge:
        if len(self.method_graphs) == 0:
            raise ValueError(
                "No method graphs found. Please trace (export) the model first."
            )
        if partitioners is None:
            edge_program: EdgeProgramManager = to_edge(self.method_graphs)
        else:
            edge_program: EdgeProgramManager = to_edge_transform_and_lower(
                self.method_graphs,
                partitioner=partitioners,
            )

        self.edge_program = edge_program

        return edge_program

    def to_executorch(self) -> ExecutorchProgram:
        if self.edge_program is None:
            raise ValueError("No edge program found. to_edge() must be called first.")

        # create the backend config:
        backend_config = ExecutorchBackendConfig(
            memory_planning_pass=SharedMemoryPlanningPass(
                shared_buffers=self.registered_shared_buffers,
                memory_planning_algo=greedy,
            )
            # emit_stacktrace=True,
        )
        # Named Buffers:
        self.executorch_program = self.edge_program.to_executorch(config=backend_config)
        # debug:
        for key, method in self.executorch_program._execution_programs.items():
            method.graph.print_tabular()
        return self.executorch_program

    def save(self, dir: Path, name: str):
        if self.executorch_program is None:
            raise ValueError(
                "No executorch program found. to_executorch() must be called first."
            )
        # create the directory if it does not exist:
        if not dir.exists():
            dir.mkdir(parents=True)
        path = dir / (name + ".pte")
        with open(path, "wb") as f:
            f.write(self.executorch_program.buffer)
