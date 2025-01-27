# file that helps generate the exporter
import copy
from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path
from types import MethodType
from typing import Callable, Optional
from operator import attrgetter
import executorch.exir
import executorch.exir.dialects
import executorch.exir.dialects.edge._ops
import executorch.exir.dialects.edge.op.api
import torch
from torch.export import (
    export,
    Dim,
    ExportedProgram,
    ExportGraphSignature,
)
import executorch
from executorch.exir import to_edge, to_edge_transform_and_lower, EdgeProgramManager
from executorch.exir import ExecutorchProgram, ExecutorchBackendConfig
from executorch.exir.pass_base import PassResult
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass, greedy
from executorch.devtools.etrecord._etrecord import generate_etrecord
from contextlib import contextmanager
from torch._dynamo import assume_constant_result
import inspect

from executorch.devtools.etrecord._etrecord import ETRecord
from executorch.util.activation_memory_profiler import generate_memory_trace
from torch.export.dynamic_shapes import _Dim

import executorch.exir.dialects.edge

import executorch.exir.dialects.edge.op

# RESERVED FUNCTION NAMES FOR SHARED BUFFERS:
# init function (used by memory planning pass)
ET_SHARED_BUFFER_INIT_FN = "et_module_init"
# constant methods to pass the shared buffer memory plan to the runtime:
ET_GET_SHARED_BUFFER_NAMES_FN = "et_get_shared_buffer_names"
ET_GET_SHARED_BUFFER_MEMORY_PLAN_FN = "et_get_shared_buffer_memory_plan"

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

        self.shared_mem_id = 2
        self.shared_buffer_size = 0

    def run(
        self,
        graph_module: torch.fx.GraphModule,
        graph_signature: Optional[ExportGraphSignature],
    ) -> PassResult:
        for node in graph_module.graph.nodes:
            if _is_buffer(node, graph_signature):
                buffer_name = graph_signature.inputs_to_buffers[node.target]
                if buffer_name in self.shared_buffers:
                    # shared mutable buffers are always mem_id 2, internal layout is done in the init_shared_buffers phase.
                    node.meta["spec"].mem_id = self.shared_mem_id
                    # this is a shared mutable buffer, its lifetime is infinite (max int64):
                    # once the memory is planned, we will update the lifetime to the actual lifetime of the buffer (max val of nodes in graph)
                    node.meta["spec"].lifetime = [0, 9223372036854775807]

        parent_result = super().run(graph_module, graph_signature)

        num_nodes = len(parent_result.graph_module.graph.nodes)

        if self.init_shared_buffers:
            # pull the buffer layout that was memory planned (we only do this once, using the et_module_init method):
            print("pulling buffer layout")
            for node in parent_result.graph_module.graph.nodes:
                if _is_buffer(node, graph_signature):
                    buffer_name = graph_signature.inputs_to_buffers[node.target]
                    if buffer_name in self.shared_buffers:
                        # this node is in our shared buffers:
                        # we need to save mem_id, mem_obj_id, and mem_offset, 
                        # as well as the actual size of the buffer:

                        num_elements = reduce(lambda x, y: x * y, node.meta["spec"].shape)
                        self.shared_buffers_memory_layout[buffer_name] = {
                            "mem_id": node.meta["spec"].mem_id,
                            "mem_obj_id": node.meta["spec"].mem_obj_id,
                            "mem_offset": node.meta["spec"].mem_offset,
                            "mem_allocated_size": node.meta["spec"].allocated_memory,
                            "mem_actual_size": num_elements * node.meta["spec"].dtype.itemsize,
                            "num_elements": num_elements,
                        }
            self.shared_buffer_size = parent_result.graph_module.meta[
                "non_const_buffer_sizes"
            ][self.shared_mem_id]
        else:
            for node in parent_result.graph_module.graph.nodes:
                if _is_buffer(node, graph_signature):
                    buffer_name = graph_signature.inputs_to_buffers[node.target]
                    if buffer_name in self.shared_buffers:
                        node.meta["spec"].lifetime = [0, num_nodes - 1]
                        # update the memory layout w/ the shared buffer memory layout:
                        node.meta["spec"].mem_id = self.shared_buffers_memory_layout[
                            buffer_name
                        ]["mem_id"]
                        node.meta["spec"].mem_obj_id = (
                            self.shared_buffers_memory_layout[buffer_name]["mem_obj_id"]
                        )
                        node.meta["spec"].mem_offset = (
                            self.shared_buffers_memory_layout[buffer_name]["mem_offset"]
                        )
                        pass
            if len(self.shared_buffers) > 0:
                parent_result.graph_module.meta["non_const_buffer_sizes"][
                    self.shared_mem_id
                ] = self.shared_buffer_size

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
    return backend_config.memory_planning_pass.shared_buffers_memory_layout

def create_shared_buffer_memory_info_constant_methods(shared_buffer_info: dict):
    # we need to create 2 shared buffer constant methods to pass the shared buffer memory info to the runtime:
    # 1. has the names of the shared buffers (ie the keys of the dicts) as c strings in a 2d byte tensor.
    # 2. has the memory info for each shared buffer as a 2d size_t tensor.

    #todo clean this up.
    def string_to_tensor(string: str):
        return torch.tensor([string.encode("utf-8")], dtype=torch.uint8)


    list_of_str_tensors = []
    list_of_mem_info_tensors = []
    for key, val in shared_buffer_info.items():
        # convert the key to a c string:
        list_of_str_tensors.append(string_to_tensor(key))
        # memory info:
        mem_info_tensor = torch.tensor([val["mem_id"], val["mem_obj_id"], val["mem_offset"], val["mem_allocated_size"], val["mem_actual_size"], val["num_elements"]], dtype=torch.long)
        list_of_mem_info_tensors.append(mem_info_tensor)
    
    # get the max length of the strings:
    max_len = max([tensor.size(1) for tensor in list_of_str_tensors])
    num_strings = len(list_of_str_tensors)
    # create a 2d tensor with the strings (add 1 for the null terminator):
    str_tensor = torch.zeros((num_strings, max_len + 1), dtype=torch.uint8)
    for i, tensor in enumerate(list_of_str_tensors):
        str_tensor[i, :tensor.size(1)] = tensor[:]

    # create a 2d tensor with the memory info:
    mem_info_tensor = torch.zeros((num_strings, 6), dtype=torch.long)
    for i, tensor in enumerate(list_of_mem_info_tensors):
        mem_info_tensor[i, :len(tensor)] = tensor
    
    return str_tensor, mem_info_tensor


def add_buffer_mutation_to_graph(
    gm: torch.fx.GraphModule, graph_signature: ExportGraphSignature, buffers: list[str]
):
    # Shared buffers must always be marked as mutable, even if they not mutated.
    # To add this in without digging (too much) through export, dynamo, tracing,
    # or exir, we do the stupid thing, and mutate all shared buffers.

    # Specifically we add a self.shared_buffer.copy_(self.shared_buffer) to the
    # begining of the graph.

    # Attempt #1
    # THIS DOES NOT WORK: (It is hard to reliably determine if a buffer is muatated).
    # This gets pulled out during the to_edge stage, with
    # _remove_unneccessary_copy_op_pass in torch.export.exported_program.py
    # The copy op is removed and the output node is mapped directly to the input
    # creating a fully functionalized graph. This allows us to do standard memory
    # planning as if all the shared buffers were mutated.

    # Attempt #2:
    # Instead of above, just mutate all shared buffers,  with the stupid copy_ op.
    # then remove the copy ops after to_edge.

    # Finally in the to_executorch stage, insert_write_back_for_buffers_pass adds
    # in copy operations to any output nodes that are buffer mutations of input
    # nodes. However, it does not add in copy operations if the copy src and
    # target are the same node.

    # The end result is that there is no additional overhead, and this
    # copy-operation is removed, but the buffers are treated otherwise as
    # mutations.

    # TODO to_edge_transform_and_lower adds some additional copy ops that seem wierd... (All this works as expected with to_edge)

    shared_buffers_in_graph = set()
    shared_buffer_input_name_to_node: dict[str, torch.fx.Node] = {}

    # go through the graph and find all the shared buffers, also identify if they are mutated.
    for node in gm.graph.nodes:
        if (
            node.op == "placeholder"
            and node.target in graph_signature.inputs_to_buffers
        ):
            shared_buffers_in_graph.add(graph_signature.inputs_to_buffers[node.target])
            shared_buffer_input_name_to_node[node.name] = node
            pass

    print(f"Shared buffers in graph: {shared_buffers_in_graph}")
    # add the copy_ node for all methods, even if they are already mutated.

    for node in gm.graph.nodes:
        if node.op != "placeholder":
            # This is the first node that is not a placeholder, insert the copy_ node before it.
            for buffer in shared_buffers_in_graph:
                with gm.graph.inserting_before(node) as insert_node:
                    # reverse lookup dict
                    buffer_input_name = [
                        k
                        for k, v in graph_signature.inputs_to_buffers.items()
                        if v == buffer
                    ][0]
                    buffer_input_node = shared_buffer_input_name_to_node[
                        buffer_input_name
                    ]
                    insert_node = gm.graph.create_node(
                        "call_function",
                        torch.ops.aten.copy_.default,
                        (buffer_input_node, buffer_input_node),
                    )
                    node.replace_input_with(buffer_input_node, insert_node)
                    # add recompile
                    gm.recompile()
                    pass
            break
        pass
    pass

    # add recompile

    print(f"graph_signature: {graph_signature}")


def remove_copy_ops_with_same_src_and_target(
    gm: torch.fx.GraphModule, graph_signature: ExportGraphSignature
):
    # Find and remove copy operations where source and target are the same
    for node in gm.graph.nodes:
        if (
            node.op == "call_function"
            # ops are edge variants:
            and hasattr(node.target, "_name")
            and hasattr(node.target, "_overloadname")
            and node.target._name == "aten::copy"
            and node.target._overloadname == "default"
        ):
            # Check if source and target tensors are the same
            src = node.args[1]  # Source tensor
            target = node.args[0]  # Target tensor

            if src == target:
                output_args = [
                    out_spec.arg.name for out_spec in graph_signature.output_specs
                ]
                # This should never be in output args - if it is, something is wrong
                if node.name in output_args:
                    raise ValueError(
                        f"Found self-copy operation '{node.name}' in graph outputs. "
                        "This indicates a problem with the graph transformation."
                    )

                # Replace all uses of this copy operation with the original tensor
                node.replace_all_uses_with(src)
                gm.graph.erase_node(node)
    gm.recompile()


##############################################
# Classes for registering methods and buffers.
##############################################
@dataclass
class MethodArg:
    example_input: torch.Tensor
    dynamic_dims: dict[int, _Dim] = field(default_factory=dict)


@dataclass
class MethodRegistration:
    fn: Callable
    kwargs: dict[str, MethodArg]


@dataclass
class SharedBufferRegistration:
    buffer: torch.Tensor


# exporter class that wraps the module, and provides methods for exporting.
# 1. Init with a module.
# 2. Register methods of the module that you want to export (including dynamic Dims).
# 2. Register torch buffers that are shared across all methods.
# 3. TODO: quantize the model.
# 4. trace the model (export)
# 5. to_edge the modeln (optionally with backend).
# 6. to_executorch the model (alongside ETRecord).
# 6. save the executorch model + ETRecord.
class MultiEntryPointExporter:
    model: torch.nn.Module
    registered_method_dict: dict[str, MethodRegistration]
    registered_shared_buffers: dict[str, SharedBufferRegistration]

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
        # validate that kwargs are a dict of MethodArg:
        for key, arg in kwargs.items():
            if not isinstance(arg, MethodArg):
                raise ValueError(
                    f"Argument of {fn.__name__} (arg name: {key}) must be a MethodArg"
                )
            if not isinstance(arg.example_input, torch.Tensor):
                raise ValueError(
                    f"Argument of {fn.__name__} (arg name: {key}.example_input) must be a torch.Tensor"
                )
            if not isinstance(arg.dynamic_dims, dict):
                raise ValueError(
                    f"Argument of {fn.__name__} (arg name: {key}.dynamic_dims) must be a dict"
                )
            for dim_idx, dim in arg.dynamic_dims.items():
                if not isinstance(dim_idx, int):
                    raise ValueError(
                        f"Argument of {fn.__name__} (arg name: {key}.dynamic_dims) must be a dict with int keys"
                    )
                if not isinstance(dim, _Dim):
                    raise ValueError(
                        f"Argument of {fn.__name__} (arg name: {key}.dynamic_dims) must be a dict with Dim values"
                    )

            # Check that the shape of the example input is not the same as the shape of the dynamic dims: (this is a bug?)
            for dim_idx, dim in arg.dynamic_dims.items():
                dynamic_min = dim.min
                dynamic_max = dim.max
                example_dim_len = arg.example_input.size(dim_idx)
                if example_dim_len == dynamic_max:
                    raise ValueError(
                        f"Example input dimension {dim_idx} length {example_dim_len} is the same as the dynamic dim max {dynamic_max}. This is not supported??? -> possible fix is to use max_dim - 1 "
                    )
            pass
        self.registered_method_dict[fn.__name__] = MethodRegistration(fn, kwargs)

    def register_shared_buffer(self, fqn: str):
        # fqn can be a string to a buffer in a model or a module.
        try:
            object = attrgetter(fqn)(self.model)
        except AttributeError:
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
            self.registered_shared_buffers[fqn] = SharedBufferRegistration(
                copy.deepcopy(object)
            )
        elif isinstance(object, torch.nn.Module):
            # add all buffers that are not marked as non persistent:
            for name, buffer in object.named_buffers():  # object is a buffer.
                if name in object.state_dict():  # buffer is persistent.
                    self.register_shared_buffer(fqn + "." + name)
        else:
            raise ValueError(
                f"register_shared_buffer: Object {fqn} in {self.model.__class__.__name__} must be a Tensor or Module"
            )

    def register_shared_buffers(self, fqn: list[str]):
        for fqn in fqn:
            self.register_shared_buffer(fqn)

    def export(self) -> dict[str, ExportedProgram]:
        if len(self.registered_shared_buffers) > 0:
            # copy this so that it can be captured by the init function:
            default_dict = {}
            for key, val in self.registered_shared_buffers.items():
                default_dict[key] = val.buffer

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
            self.registered_method_dict["et_module_init"] = MethodRegistration(
                self.model.et_module_init,
                {},
            )
            # # et_module_init(self.model)

        with torch.no_grad():
            for method in self.registered_method_dict:
                method_registration = self.registered_method_dict[method]
                # update the forward method of the model:
                with patch_forward(self.model, method_registration.fn):
                    sig = inspect.signature(self.model.forward)
                    param_names = [param.name for param in sig.parameters.values()]
                    print(param_names)
                    example_args = {}
                    dynamic_shapes = {}
                    # check that all registered parameters are present in the method signature:
                    for registered_kwarg in method_registration.kwargs:
                        if registered_kwarg not in param_names:
                            raise ValueError(
                                f"Parameter {registered_kwarg} not found in function {method_registration.fn.__name__} signature, options are {param_names}"
                            )
                    for param in param_names:
                        if param not in method_registration.kwargs:
                            raise ValueError(
                                f"Parameter {param} not found in function {method_registration.fn.__name__} registration, options are {method_registration.kwargs.keys()}"
                            )
                        else:
                            param_value = method_registration.kwargs[param]
                            # parameter has been registered:
                            example_args[param] = param_value.example_input
                            dynamic_shapes[param] = param_value.dynamic_dims

                    method_graph: ExportedProgram = export(
                        self.model,
                        (),
                        kwargs=example_args,
                        dynamic_shapes=dynamic_shapes,
                        strict=True,
                    )
                    print("--------------------------------")
                    print(f"Method Graph: {method}")
                    add_buffer_mutation_to_graph(
                        method_graph.graph_module,
                        method_graph.graph_signature,
                        self.registered_shared_buffers,
                    )
                    self.method_graphs[method] = method_graph
            print("--------------------------------")
            print("Method Graphs:")
            for key, method in self.method_graphs.items():
                print("  ------------------------------")
                print(f"  method: {key}")
                print("  " + str(method.graph))

        return self.method_graphs

    # def quantize_model(self):
    #     # check that we have a model traced (model graphs not empty):
    #     if len(self.method_graphs) == 0:
    #         raise ValueError("No method graphs found. Please trace the model first.")
    #     pass

    def to_edge(
        self, constant_methods: dict =None, partitioners: list = None
    ) -> EdgeProgramManager:
        # export the model graphs to the edge:
        if len(self.method_graphs) == 0:
            raise ValueError(
                "No method graphs found. Please trace (export) the model first."
            )

        edge_program: EdgeProgramManager = to_edge(
            self.method_graphs,
            constant_methods=constant_methods,
        )

        for method in edge_program._edge_programs:
            remove_copy_ops_with_same_src_and_target(
                edge_program._edge_programs[method].graph_module,
                edge_program._edge_programs[method].graph_signature,
            )

        self.edge_program = edge_program

        # validate that the edge program is valid:
        print("--------------------------------")
        print("Edge Program:")
        for key, method in self.edge_program._edge_programs.items():
            print("  ------------------------------")
            print(f"  method: {key}")
            print("  " + str(method.graph))

        return edge_program

    def to_executorch(self) -> ExecutorchProgram:
        if self.edge_program is None:
            raise ValueError("No edge program found. to_edge() must be called first.")



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
        shared_layout_dict = initialize_shared_memory_planning(backend_config, self.edge_program)

        # hack this in as an additional constant_method to pass memory planning data to the runtime for
        name_tensor, memory_plan_tensor = create_shared_buffer_memory_info_constant_methods(shared_layout_dict)
        if not self.edge_program._config_methods:
            self.edge_program._config_methods = {}
        self.edge_program._config_methods[ET_GET_SHARED_BUFFER_NAMES_FN] = name_tensor
        self.edge_program._config_methods[ET_GET_SHARED_BUFFER_MEMORY_PLAN_FN] = memory_plan_tensor
        
        # deepcopy the edge program for later use
        self.edge_program_copy = copy.deepcopy(self.edge_program)
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
        mk_subdir: bool = True,
    ):
        model_name = self.model.__class__.__name__
        if self.executorch_program is None:
            raise ValueError(
                "No executorch program found. to_executorch() must be called first."
            )
        # create the directory if it does not exist:
        if not dir.exists():
            dir.mkdir(parents=True)
        if mk_subdir:
            subdir = dir / model_name
            subdir.mkdir(parents=True, exist_ok=True)
            dir = subdir
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

        if op_trace:
            for method in self.executorch_program.methods:
                ## Save the executorch graph trace:
                output_file = dir / f"{name}-{method}-graph_trace_executorch.txt"
                with open(output_file, "w") as f:
                    f.write(str(self.executorch_program.exported_program(method).graph))
                output_file = dir / f"{name}-{method}-graph_trace_executorch_with_stack.txt"
                with open(output_file, "w") as f:
                    f.write(str(self.executorch_program.exported_program(method)))

                ## Save the edge graph trace:
                output_file = dir / f"{name}-{method}-graph_trace_edge.txt"
                with open(output_file, "w") as f:
                    f.write(str(self.edge_program.exported_program(method).graph))
                output_file = dir / f"{name}-{method}-graph_trace_edge_with_stack.txt"
                with open(output_file, "w") as f:
                    f.write(str(self.edge_program.exported_program(method)))

                ## Save the exported graph trace:
                output_file = dir / f"{name}-{method}-graph_trace_exported.txt"
                with open(output_file, "w") as f:
                    f.write(str(self.method_graphs[method].graph))
                output_file = dir / f"{name}-{method}-graph_trace_exported_with_stack.txt"
                with open(output_file, "w") as f:
                    f.write(str(self.method_graphs[method]))

        if memory_trace:
            for method in self.executorch_program.methods:
                output_file = dir / f"{name}-{method}-memory_profile.json"
                generate_memory_trace(
                    executorch_program_manager=self.executorch_program,
                    chrome_trace_filename=output_file,
                    enable_memory_offsets=True,
                    method_name=method,
                )
