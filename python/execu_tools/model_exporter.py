# file that helps generate the exporter
import copy
from enum import Enum
from pathlib import Path
from types import MethodType
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
from torch._dynamo import assume_constant_result
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
        init_shared_buffers: bool = False,
        **kwargs,
    ):
        self.shared_buffers = shared_buffers
        self.init_shared_buffers = init_shared_buffers
        self.shared_buffers_memory_layout = {}
        super().__init__(**kwargs)

    def run(
        self,
        graph_module: torch.fx.GraphModule,
        graph_signature: Optional[ExportGraphSignature],
    ) -> PassResult:
        for node in graph_module.graph.nodes:
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

        if self.init_shared_buffers:
            # pull the buffer layout that was memory planned:
            print("pulling buffer layout")
            for node in parent_result.graph_module.graph.nodes:
                if _is_buffer(node, graph_signature):
                    buffer_name = graph_signature.inputs_to_buffers[node.target]
                    if buffer_name in self.shared_buffers:
                        # this node is in our shared buffers:
                        # we need to save mem_id, mem_obj_id, and mem_offset:
                        self.shared_buffers_memory_layout[buffer_name] = {
                            "mem_id": node.meta["spec"].mem_id,
                            "mem_obj_id": node.meta["spec"].mem_obj_id,
                            "mem_offset": node.meta["spec"].mem_offset,
                        }

        for node in parent_result.graph_module.graph.nodes:
            if _is_buffer(node, graph_signature):
                buffer_name = graph_signature.inputs_to_buffers[node.target]
                if buffer_name in self.shared_buffers:
                    node.meta["spec"].lifetime = [0, num_nodes - 1]
                    # update the memory layout w/ the shared buffer memory layout:
                    node.meta["spec"].mem_id = self.shared_buffers_memory_layout[
                        buffer_name
                    ]["mem_id"]
                    node.meta["spec"].mem_obj_id = self.shared_buffers_memory_layout[
                        buffer_name
                    ]["mem_obj_id"]
                    node.meta["spec"].mem_offset = self.shared_buffers_memory_layout[
                        buffer_name
                    ]["mem_offset"]
                    pass

        # we need to go back through and

        # TODO: gathar diagnostic info/ validate consistancy across all methods
        return parent_result


def initialize_shared_memory_planning(
    backend_config: ExecutorchBackendConfig, edge_program: EdgeProgramManager
):
    # validate that the memory planning pass is a shared memory planning pass:
    if not isinstance(backend_config.memory_planning_pass, SharedMemoryPlanningPass):
        raise ValueError("Memory planning pass is not a shared memory planning pass.")

    # run the memory planning pass to init shared buffers:
    tmp_edge_program = copy.deepcopy(edge_program)

    # turn on the memory planning pass init shared buffers:
    backend_config.memory_planning_pass.init_shared_buffers = True
    # only run et_module_init:
    tmp_edge_program._edge_programs = {
        name: prog
        for name, prog in tmp_edge_program._edge_programs.items()
        if name == "et_module_init"
    }
    tmp_edge_program.to_executorch(config=backend_config)

    # turn off the memory planning pass init shared buffers:
    backend_config.memory_planning_pass.init_shared_buffers = False


def add_buffer_mutation_to_graph(
    gm: torch.fx.GraphModule, graph_signature: ExportGraphSignature, buffers: list[str]
):
    # Shared buffers must always be marked as mutable, even if they not mutated.
    # To add this in without digging (too much) through export, dynamo, tracing,
    # or exir, we do the stupid thing, and mutate all shared buffers.
    # Specifically we add a self.shared_buffer.copy_(self.shared_buffer) to the 
    # begining of the graph.

    # This gets pulled out during the to_edge stage, with
    # _remove_unneccessary_copy_op_pass in torch.export.exported_program.py
    # The copy op is removed and the output node is mapped directly to the input
    # creating a fully functionalized graph. This allows us to do standard memory
    # planning as if all the shared buffers were mutated.

    # Finally in the to_executorch stage, insert_write_back_for_buffers_pass adds 
    # in copy operations to any output nodes that are buffer mutations of input 
    # nodes. However, it does not add in copy operations if the copy src and 
    # target are the same node.

    # The end result is that there is no additional overhead, and this 
    # copy-operation is removed, but the buffers are treated otherwise as 
    # mutations.

    #TODO to_edge_transform_and_lower adds some additional copy ops that seem wierd... (All this works as expected with to_edge)

    shared_buffers_in_graph = set()
    shared_buffers_mutated = set() # mutated buffers will have a copy_ node with their buffer as the target.
    shared_buffer_input_name_to_node : dict[str, torch.fx.Node] = {}


    # go through the graph and find all the shared buffers, also identify if they are mutated.
    for node in gm.graph.nodes:
        if node.op == "placeholder" and  node.target in graph_signature.inputs_to_buffers:
            shared_buffers_in_graph.add(graph_signature.inputs_to_buffers[node.target])
            shared_buffer_input_name_to_node[node.name] = node
            pass
        if node.op == "call_function" and node.target == torch.ops.aten.copy_.default:
            copy_target = node.args[0]
            if copy_target.name in graph_signature.inputs_to_buffers and graph_signature.inputs_to_buffers[copy_target.name] in shared_buffers_in_graph:
                shared_buffers_mutated.add(graph_signature.inputs_to_buffers[copy_target.name])

    
    print(f"Shared buffers in graph: {shared_buffers_in_graph}")
    print(f"Shared buffers mutated: {shared_buffers_mutated}")
    # shared buffers that are not mutated:
    shared_buffers_not_mutated = shared_buffers_in_graph - shared_buffers_mutated
    print(f"Shared buffers not mutated: {shared_buffers_not_mutated}")

    # add the copy_ node for the shared buffers that are not mutated:

    for node in gm.graph.nodes:
        if node.op != "placeholder":
            # This is the first node that is not a placeholder, insert the copy_ node before it.
            for buffer in shared_buffers_not_mutated:
                with gm.graph.inserting_before(node) as insert_node:
                    # reverse lookup dict
                    buffer_input_name = [k for k, v in graph_signature.inputs_to_buffers.items() if v == buffer][0]
                    buffer_input_node = shared_buffer_input_name_to_node[buffer_input_name]
                    insert_node = gm.graph.call_function(torch.ops.aten.copy_.default, (buffer_input_node, buffer_input_node))
                    node.replace_input_with(buffer_input_node, insert_node)
                    #add recompile?
                    pass
            break
        pass
    # add the copy_ node to the graph:
    pass


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
        if getattr(self.model.__class__, fn.__name__, None) is None:
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
            if fqn not in [
                n for n, _ in self.model.named_buffers()
            ]:  # object is a buffer.
                raise ValueError(
                    f"register_shared_buffer: {fqn} is not a buffer in {self.model.__class__.__name__}"
                )
            if fqn not in [n for n in self.model.state_dict()]:  # buffer is persistent.
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

    def export(self) -> dict[str, ExportedProgram]:
        if len(self.registered_shared_buffers) > 0:
            # copy this so that it can be captured by the init function:
            default_dict = {}
            for key, val in self.registered_shared_buffers.items():
                default_dict[key] = val[1]

            self.model.__et_export_shared_buffers_defaults = default_dict

            # create a new init function that sets all of the shared buffers:
            # @assume_constant_result
            # def _get_constant_default(tensor_name: str):
            #     value = default_dict[tensor_name]
            #     return default_dict[tensor_name]
            # TODO: Allow for non-zero initialization of shared buffers.
            def et_module_init(self_module: torch.nn.Module):
                for key in default_dict:
                    # buffer = getattr(self_module,key)
                    buffer = attrgetter(key)(self_module)
                    # const_vals = _get_constant_default(key)
                    buffer.copy_(torch.zeros_like(buffer))
                return None

            # add method to the model:
            self.model.et_module_init = MethodType(et_module_init, self.model)
            # add to method dict:
            self.registered_method_dict["et_module_init"] = (
                self.model.et_module_init,
                {},
            )
            # # et_module_init(self.model)

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
                        strict=True,
                    )
                    print(f"--------------------------------")
                    print(f"Method Graph: {fn.__name__}")
                    add_buffer_mutation_to_graph(method_graph, method_graph.graph_signature, self.registered_shared_buffers)
                    self.method_graphs[fn.__name__] = method_graph
            print(f"--------------------------------")
            print(f"Method Graphs:")
            for key, method in self.method_graphs.items():
                print(f"  ------------------------------")
                print(f"  method: {key}")
                print("  " + str(method.graph))

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

        edge_program: EdgeProgramManager = to_edge(
            self.method_graphs,
        )

        self.edge_program = edge_program

        # validate that the edge program is valid:
        print(f"--------------------------------")
        print(f"Edge Program:")
        for key, method in self.edge_program._edge_programs.items():
            print(f"  ------------------------------")
            print(f"  method: {key}")
            print("  " + str(method.graph))

        return edge_program

    def to_executorch(self) -> ExecutorchProgram:
        if self.edge_program is None:
            raise ValueError("No edge program found. to_edge() must be called first.")

        # deepcopy the edge program for later use
        self.edge_program_copy = copy.deepcopy(self.edge_program)

        # create a shared memory planning pass:
        shared_memory_planning_pass = SharedMemoryPlanningPass(
            init_shared_buffers=False,
            shared_buffers=self.registered_shared_buffers,
            memory_planning_algo=greedy,
            alloc_graph_input=False,
            alloc_graph_output=True,
        )

        # create the backend config:
        backend_config = ExecutorchBackendConfig(
            memory_planning_pass=shared_memory_planning_pass
            # emit_stacktrace=True, #does not work?
        )

        # initialize the shared buffers (this will set the memory layout of the shared buffers):
        initialize_shared_memory_planning(backend_config, self.edge_program)

        # Export for real (now that we have the shared buffer memory layout):
        self.executorch_program = self.edge_program.to_executorch(config=backend_config)

        # debug:
        for key, method in self.executorch_program._execution_programs.items():
            print(f"method: {key}")
            print(method.graph)
        return self.executorch_program

    def save(
        self,
        dir: Path,
        name: str,
        et_record: bool = True,
        memory_trace: bool = True,
        op_trace=True,
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
            # TODO fix the requirement that the edge program is not None.
            generate_etrecord(
                save_path,
                None,
                self.executorch_program,
                {model_name: self.edge_program_copy},
            )

        if op_trace:
            for method in self.executorch_program.methods:
                output_file = dir / f"{model_name}-{method}-graph_trace.txt"
                with open(output_file, "w") as f:
                    f.write(str(self.executorch_program.exported_program(method).graph))

        if memory_trace:
            for method in self.executorch_program.methods:
                output_file = dir / f"{model_name}-{method}-memory_profile.json"
                generate_memory_trace(
                    executorch_program_manager=self.executorch_program,
                    chrome_trace_filename=output_file,
                    enable_memory_offsets=True,
                    method_name=method,
                )
