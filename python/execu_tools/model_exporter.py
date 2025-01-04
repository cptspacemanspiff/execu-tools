# file that helps generate the exporter
from enum import Enum
from pathlib import Path
import torch
from torch.export import export, export_for_training, ExportedProgram
from executorch.exir import to_edge, to_edge_transform_and_lower, EdgeProgramManager
from executorch.exir import ExecutorchProgram, ExecutorchBackendConfig
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from executorch.exir.memory_planning import materialize_buffer

from contextlib import contextmanager

import inspect


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

    method_graphs: dict[str, ExportedProgram]

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.registered_method_dict = {}

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
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False, alloc_graph_output=False
            ),
            # emit_stacktrace=True,
        )

        self.executorch_program = self.edge_program.to_executorch(config=backend_config)
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
