# file that helps generate the exporter
import copy
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Union
from operator import attrgetter
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
from executorch.devtools.etrecord._etrecord import generate_etrecord
from contextlib import contextmanager

import inspect

from executorch.exir.dynamic_shape import DynamicMemoryPlanningMode

from executorch.exir.tensor import ALIGNMENT

from executorch.devtools.etrecord._etrecord import ETRecord

from executorch.util.activation_memory_profiler import generate_memory_trace


@contextmanager
def patch_forward(obj: torch.nn.Module, new_method):
    """
    Patches the forward method of a PyTorch module, needs to patch the class not
    just the instance b/c otherwise export will error out when trying to produce
    items that reference self.
    """
    original_method = obj.__class__.forward
    # obj.__class__.forward = new_method
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
                        # once the memory is planned, we will update the lifetime to the actual lifetime of the buffer (max val of nodes in graph)
                        node.meta["spec"].lifetime = [0, 9223372036854775807]
                        # this must be a shared mutable buffer, even if we do not modify it, treat it as non-const.
                        # it will not get correctly if we do not treat it as non-const and add it to buffers to mutate.
                        # node.meta["spec"].const = False
                        # graph_module.graph.buffers_to_mutate.add(node.target)

        parent_result = super().run(graph_module, graph_signature)

        num_nodes = len(parent_result.graph_module.graph.nodes)

        for subgm in parent_result.graph_module.modules():
            if not isinstance(subgm, torch.fx.GraphModule):
                continue
            for node in subgm.graph.nodes:
                if _is_buffer(node, graph_signature):
                    buffer_name = graph_signature.inputs_to_buffers[node.target]
                    if buffer_name in self.shared_buffers:
                        node.meta["spec"].lifetime = [0, num_nodes - 1]
                        # add this to the mutable buffer list so it will be placed correctly:

        # we need to go back throuj and

        # TODO: gathar diagnostic info/ validate consistancy across all methods
        print("done")
        return parent_result


# exporter class that wraps the module, and provides methods for exporting.
# 1. Init with a module.
# 2. Register methods of the module that you want to export (including dynamic Dims).
# 2. Regsister torch buffers that are shared across all methods.
# 3. TODO: quantize the model.
# 4. trace the model (export)
# 5. to_edge the modeln (optionally with backend).
# 6. to_executorch the model (alongside ETRecord).
# 6. save the executorch model + ETRecord.
class Exporter:
    model: torch.nn.Module
    registered_method_dict: dict[str, tuple[callable, dict]]
    registered_shared_buffers: dict[str, tuple[torch.Tensor, Callable]]

    method_graphs: dict[str, ExportedProgram]

    et_record: ETRecord

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.registered_method_dict = {}
        self.registered_shared_buffers = {}
        # intermediary states:
        # self.quantized_model  :dict[str, ExportedProgram] = None #from quantize
        self.method_graphs: dict[str, ExportedProgram] = {}  # from export
        self.edge_program: EdgeProgramManager = None  # from to_edge
        self.edge_program_copy: EdgeProgramManager = (
            None  # from to_executorch (used in save, for etrecord)
        )
        self.executorch_program: ExecutorchProgram = None  # from to_executorch

    def register(self, fn, **kwargs):
        # validate that the function is a method of the module.
        if fn.__qualname__.split(".")[0] != self.model.__class__.__name__:
            raise ValueError(
                f"Function {fn.__name__} is not a method of {self.model.__class__.__name__}"
            )
        self.registered_method_dict[fn.__name__] = (fn, kwargs)

    def register_shared_buffer(self, fqn: str):
        # fqn can be a string to a buffer in a model or a module.
        try:
            object = attrgetter(fqn)(self.model)
        except AttributeError as e:
            raise ValueError(
                f"register_shared_buffer: Object {fqn} does not exist in {self.model.__class__.__name__}"
            )
        if isinstance(object, torch.Tensor):
            # check if the object is a buffer:
            if not fqn in [
                n for n, _ in self.model.named_buffers()
            ]:  # object is a buffer.
                raise ValueError(
                    f"register_shared_buffer: {fqn} is not a buffer in {self.model.__class__.__name__}"
                )
            if not fqn in [n for n in self.model.state_dict()]:  # buffer is persistent.
                raise ValueError(
                    f"register_shared_buffer: Buffer {fqn} is not persistent in {self.model.__class__.__name__}"
                )
            # copy the buffer to avoid touching the original buffer, when it is used for initialization.
            self.registered_shared_buffers[fqn] = copy.deepcopy(object)
        elif isinstance(object, torch.nn.Module):
            # add all buffers that are not marked as non persistent:
            for name, buffer in object.named_buffers():  # object is a buffer.
                if name in object.state_dict():  # buffer is persistent.
                    self.register_shared_buffer(fqn + "." + name)
        else:
            raise ValueError(
                f"register_shared_buffer: Object {fqn} in {self.model.__class__.__name__} must be a Tensor or Module"
            )

    # def register_shared_buffer(self, fqn: str):
    #     if isinstance(fqn, str):
    #         # adding via fqn:
    #         # validate that the buffer exists in the model state_dict:
    #         if not fqn in self.model.state_dict():
    #             raise ValueError(
    #                 f"Buffer {fqn} does not exist in {self.model.__class__.__name__} state dict."
    #             )
    #         self.registered_shared_buffers.add(fqn)
    #     else:
    #         raise ValueError(f"Cannot register buffer of type {type(buffer)}")

    # def register_shared_buffer(self, module: torch.nn.Module, recursive: bool = True):
    #     if isinstance(module, torch.nn.Module):
    #         # adding via object, add all buffers not marked as non persistent:
    #         for _name, _buffer in module.named_buffers(recursive=recursive):
    #             if _name in module.state_dict().keys():
    #                 self.registered_shared_buffers.add(_name)

    def export(self) -> dict[str, ExportedProgram]:

        if len(self.registered_shared_buffers) > 0:
            # create a new init function that sets all of the shared buffers:
            def et_module_init(module : torch.nn.Module):
                for fqn, buffer_value in self.registered_shared_buffers.items():
                    setattr(self, fqn, torch.zeros_like(buffer_value))
                    print(f"set {fqn} to {buffer_value}")

            # add to method dict:
            self.registered_method_dict['et_module_init'] = (et_module_init, {})
            # et_module_init(self.model)

        with torch.no_grad():
            for method in self.registered_method_dict:
                fn, kwargs = self.registered_method_dict[method]
                # update the forward method of the model:
                with patch_forward(self.model, fn):
                    sig = inspect.signature(self.model.set_cache)
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
                        fn = self.model.set_cache,
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

        # validate that the edge program is valid:
        for key, method in self.edge_program._edge_programs.items():
            method.graph.print_tabular()

        return edge_program

    def to_executorch(self) -> ExecutorchProgram:
        if self.edge_program is None:
            raise ValueError("No edge program found. to_edge() must be called first.")

        # deepcopy the edge program for later use
        self.edge_program_copy = copy.deepcopy(self.edge_program)

        # create the backend config:
        backend_config = ExecutorchBackendConfig(
            memory_planning_pass=SharedMemoryPlanningPass(
                shared_buffers=self.registered_shared_buffers,
                memory_planning_algo=greedy,
                alloc_graph_input=False,
                alloc_graph_output=True,
            ),
            # emit_stacktrace=True, #does not work?
        )
        # Does this mutate the edge program?
        self.executorch_program = self.edge_program.to_executorch(config=backend_config)
        # debug:
        for key, method in self.executorch_program._execution_programs.items():
            print(f"method: {key}")
            nodes = method.graph_module.graph.nodes
            method.graph.print_tabular()
            # for node in method.graph_module
        return self.executorch_program

    def save(
        self, dir: Path, name: str, et_record: bool = True, memory_trace: bool = True
    ):
        model_name = self.model.__class__.__name__
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

        if et_record:
            save_path = dir / (name + ".etrecord")
            generate_etrecord(
                save_path,
                None,
                self.executorch_program,
                {model_name: self.edge_program_copy},
            )

        if memory_trace:
            for method in self.executorch_program.methods:
                output_file = dir / f"{model_name}-{method}-memory_profile.json"
                generate_memory_trace(
                    executorch_program_manager=self.executorch_program,
                    chrome_trace_filename=output_file,
                    enable_memory_offsets=True,
                    method_name=method,
                )
