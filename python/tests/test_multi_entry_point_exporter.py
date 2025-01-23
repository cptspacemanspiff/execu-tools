import pytest
from pathlib import Path
import torch
from torch.export import Dim
from execu_tools.model_exporter import MethodArg, MultiEntryPointExporter

class Submodule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("submodule_buffer1", torch.ones(10, 20), persistent=True)
        self.register_buffer("submodule_buffer2", torch.ones(10, 20), persistent=True)
        self.register_buffer("submodule_non_persistent_buffer", torch.ones(10, 20), persistent=False)

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("buffer1", torch.ones(10, 20), persistent=True)
        self.register_buffer("buffer2", torch.ones(10, 20), persistent=True)
        self.register_buffer("non-persistent-buffer", torch.ones(10, 20), persistent=False)

        self.value = 10
        self.tensor = torch.ones(10, 20)
        self.param = torch.nn.Parameter(torch.ones(10, 20))

        self.submodule = Submodule()

        self.other_submodule = Submodule()

    def method1(self, x: torch.Tensor):
        return x + self.other_submodule.submodule_buffer2

    def method2(self, x: torch.Tensor):
        self.buffer1.copy_(x)
        return None

    def method3(self, x: torch.Tensor, y: torch.Tensor):
        return x + y

    def load_from_buffer1(self, x: torch.Tensor):
        x.copy_(self.buffer1)
        return None

    def get_buffer1(self):
        return self.buffer1
    
    def set_buffer1(self, x: torch.Tensor):
        self.buffer1.copy_(x)
        return None

    def set_buffer_dynamic(self, x: torch.Tensor):
        narrowed_buffer_0 = self.buffer1.narrow(0, 0, x.shape[0])
        narrowed_buffer_1 = narrowed_buffer_0.narrow(1, 0, x.shape[1])
        narrowed_buffer_1.copy_(x)
        return None
    
    def load_from_buffer_dynamic(self, x: torch.Tensor):
        x.copy_(self.buffer1.narrow(0, 0, x.size(0)).narrow(1, 0, x.size(1)))
        return None


@pytest.fixture
def model():
    return SimpleModel()


@pytest.fixture
def exporter(model):
    return MultiEntryPointExporter(model)


def test_register_method(exporter: MultiEntryPointExporter, model: SimpleModel):
    """Test registering a method with the exporter"""

    test_dim_0 = Dim("dim0", min=1, max=10)

    exporter.register(model.method1, x=MethodArg(torch.ones(10, 20)))
    exporter.register(model.method2, x=MethodArg(torch.ones(10, 20), {0: test_dim_0}))

    # Assert method1 is registered
    assert "method1" in exporter.registered_method_dict
    assert exporter.registered_method_dict["method1"].fn == model.method1
    registered_arg1 = exporter.registered_method_dict["method1"].kwargs["x"]
    assert torch.equal(registered_arg1.example_input, torch.ones(10, 20))
    assert registered_arg1.dynamic_dims == {}

    # Assert method2 is registered
    assert "method2" in exporter.registered_method_dict
    assert exporter.registered_method_dict["method2"].fn == model.method2
    registered_arg2 = exporter.registered_method_dict["method2"].kwargs["x"]
    assert torch.equal(registered_arg2.example_input, torch.ones(10, 20))
    assert registered_arg2.dynamic_dims == {0: test_dim_0}


def test_register_invalid_method_arg(exporter: MultiEntryPointExporter, model: SimpleModel):
    """Test registering a method with invalid arguments"""
    
    # Test non-MethodArg argument
    with pytest.raises(ValueError):
        exporter.register(model.method1, x=torch.ones(1))

    # Test non-Tensor example_input
    with pytest.raises(ValueError):
        exporter.register(model.method1, x=MethodArg(example_input=(torch.ones(1),)))

    # Test non-dict dynamic_dims
    with pytest.raises(ValueError):
        exporter.register(model.method1, x=MethodArg(torch.ones(1), dynamic_dims=[]))

    # Test dynamic_dims with non-int keys
    with pytest.raises(ValueError):
        exporter.register(model.method1, x=MethodArg(torch.ones(1), dynamic_dims={"0": Dim("dim0", min=1, max=10)}))

    # Test dynamic_dims with non-Dim values
    with pytest.raises(ValueError):
        exporter.register(model.method1, x=MethodArg(torch.ones(1), dynamic_dims={0: "not_a_dim"}))

def test_register_invalid_method(exporter: MultiEntryPointExporter):
    """Test registering a method that doesn't exist"""

    def invalid_method(x):
        return x

    with pytest.raises(ValueError):
        exporter.register(invalid_method, x=MethodArg(torch.ones(1)))

def test_register_shared_buffer_by_fqn(exporter: MultiEntryPointExporter, model: SimpleModel):
    """Test registering a shared buffer by fqn"""

    # register a buffer that does not exist
    with pytest.raises(ValueError):
        exporter.register_shared_buffer("non-existent-buffer")

    # register a buffer that exists
    exporter.register_shared_buffer("buffer1")
    assert "buffer1" in exporter.registered_shared_buffers

    # fail to register non-persistent buffer
    with pytest.raises(ValueError):
        exporter.register_shared_buffer("non-persistent-buffer")

    # register a buffer that exists in a submodule
    exporter.register_shared_buffer("submodule.submodule_buffer1")
    assert "submodule.submodule_buffer1" in exporter.registered_shared_buffers

    # register a non-persistent buffer in a submodule (fails)
    with pytest.raises(ValueError):
        exporter.register_shared_buffer("submodule.submodule_non_persistent_buffer")

    # register a whole submodule
    exporter.register_shared_buffer("other_submodule")
    assert "other_submodule.submodule_buffer1" in exporter.registered_shared_buffers
    assert "other_submodule.submodule_buffer2" in exporter.registered_shared_buffers
    
    # fail to register a non-buffer attribute
    with pytest.raises(ValueError):
        exporter.register_shared_buffer("other_submodule.value")
    
    # fail to register a non-buffer parameter
    with pytest.raises(ValueError):
        exporter.register_shared_buffer("other_submodule.param")

    # fail to register a non-buffer tensor
    with pytest.raises(ValueError):
        exporter.register_shared_buffer("other_submodule.tensor")

def test_export_single_method(exporter: MultiEntryPointExporter, model: SimpleModel):
    """Test exporting a single method"""
    exporter.register(model.method1, x=MethodArg(torch.ones(10, 20)))
    method_graphs = exporter.export()
    assert "method1" in method_graphs

    # test that exporting a method with missing arguments fails
    with pytest.raises(ValueError):
        exporter.register(model.method1)
        exporter.export()

    # test that exporting a method with additional (unused) arguments fails
    with pytest.raises(ValueError):
        exporter.register(model.method1, x=MethodArg(torch.ones(10, 20)), y=MethodArg(torch.ones(10, 20)))
        exporter.export()

def test_export_multiple_methods(exporter: MultiEntryPointExporter, model: SimpleModel):
    """Test exporting multiple methods"""
    exporter.register(model.method1, x=MethodArg(torch.ones(10, 20)))
    exporter.register(model.method2, x=MethodArg(torch.ones(10, 20)))
    method_graphs = exporter.export()
    assert "method1" in method_graphs
    assert "method2" in method_graphs

def test_export_with_shared_buffers(exporter: MultiEntryPointExporter, model: SimpleModel):
    """Test exporting methods with shared buffers"""
    exporter.register_shared_buffer("buffer1")
    exporter.register(model.method2, x=MethodArg(torch.ones(10, 20)))
    method_graphs = exporter.export()
    assert "method2" in method_graphs
    assert "et_module_init" in method_graphs  # Should create init method

def test_copy_insertion(exporter: MultiEntryPointExporter, model: SimpleModel):
    """Test that copy operations are inserted for shared buffers that are not mutated"""
    model = SimpleModel()
    exporter = MultiEntryPointExporter(model)

    # Register buffer1 as a shared buffer
    exporter.register_shared_buffer("buffer1")
    
    # Register methods with different buffer usage patterns:
    # - get_buffer1: only reads buffer1 without mutation
    # - set_buffer1: explicitly mutates buffer1
    # - load_from_buffer1: reads buffer1 and mutates the input tensor
    exporter.register(model.get_buffer1)
    exporter.register(model.set_buffer1, x=MethodArg(torch.ones(10, 20)))
    exporter.register(model.load_from_buffer1, x=MethodArg(torch.ones(10, 20)))
    
    method_graphs = exporter.export()
    
    # Check that all methods are exported
    assert "get_buffer1" in method_graphs
    assert "set_buffer1" in method_graphs
    assert "load_from_buffer1" in method_graphs
    assert "et_module_init" in method_graphs  # Should create init method
    
    # Check that get_buffer1 has a copy_ operation added since it only reads
    get_buffer_graph = method_graphs["get_buffer1"]
    has_buffer_self_copy = False
    for node in get_buffer_graph.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.copy_.default:
            # Check that both source and target are the same buffer
            if (node.args[0].name in get_buffer_graph.graph_signature.inputs_to_buffers and 
                node.args[1].name in get_buffer_graph.graph_signature.inputs_to_buffers and
                get_buffer_graph.graph_signature.inputs_to_buffers[node.args[0].name] == "buffer1" and
                get_buffer_graph.graph_signature.inputs_to_buffers[node.args[1].name] == "buffer1"):
                has_buffer_self_copy = True
                break
    assert has_buffer_self_copy, "No self-copy operation found for buffer1 in get_buffer1 method"
    
    # Check that set_buffer1 has exactly one copy operation (the explicit mutation)
    set_buffer_graph = method_graphs["set_buffer1"]
    copy_op_count = 0
    for node in set_buffer_graph.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.copy_.default:
            copy_op_count += 1
            # Verify this is copying from input to buffer
            assert node.args[0].name in set_buffer_graph.graph_signature.inputs_to_buffers, "Copy target should be buffer"
            assert node.args[1].name not in set_buffer_graph.graph_signature.inputs_to_buffers, "Copy source should be input"
    assert copy_op_count == 1, "Expected exactly one copy operation in set_buffer1 method"

    # Check that load_from_buffer1 has a copy_ operation added
    # Even though it has an explicit copy_, it's copying TO the input tensor
    # We still need a copy_ for the buffer itself
    load_buffer_graph = method_graphs["load_from_buffer1"]
    buffer_copy_count = 0
    buffer_self_copy_found = False
    for node in load_buffer_graph.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.copy_.default:
            buffer_copy_count += 1
            # Check if this is a buffer self-copy
            if (node.args[0].name in load_buffer_graph.graph_signature.inputs_to_buffers and 
                node.args[1].name in load_buffer_graph.graph_signature.inputs_to_buffers and
                load_buffer_graph.graph_signature.inputs_to_buffers[node.args[0].name] == "buffer1" and
                load_buffer_graph.graph_signature.inputs_to_buffers[node.args[1].name] == "buffer1"):
                buffer_self_copy_found = True
    assert buffer_copy_count == 2, "Expected two copy operations in load_from_buffer1 method"
    assert buffer_self_copy_found, "No self-copy operation found for buffer1 in load_from_buffer1 method"

def test_copy_insertion_dynamic(exporter: MultiEntryPointExporter, model: SimpleModel):
    """Test that copy operations are inserted correctly for dynamic buffer operations"""
    # Register buffer1 as a shared buffer
    exporter.register_shared_buffer("buffer1")
    
    # Register methods with dynamic buffer operations
    test_dim_0 = Dim("dim0", min=1, max=10)
    test_dim_1 = Dim("dim1", min=1, max=20)
    
    exporter.register(
        model.set_buffer_dynamic, 
        x=MethodArg(torch.ones(5, 20), dynamic_dims={0: test_dim_0, 1: test_dim_1})
    )
    exporter.register(
        model.load_from_buffer_dynamic,
        x=MethodArg(torch.ones(5, 20), dynamic_dims={0: test_dim_0, 1: test_dim_1})
    )
    
    method_graphs = exporter.export()
    
    # Check set_buffer_dynamic method
    set_buffer_graph = method_graphs["set_buffer_dynamic"]
    set_buffer_copy_count = 0
    set_buffer_self_copy_found = False
    
    for node in set_buffer_graph.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.copy_.default:
            set_buffer_copy_count += 1
            # Check if this is a buffer self-copy
            if (node.args[0].name in set_buffer_graph.graph_signature.inputs_to_buffers and 
                node.args[1].name in set_buffer_graph.graph_signature.inputs_to_buffers and
                set_buffer_graph.graph_signature.inputs_to_buffers[node.args[0].name] == "buffer1" and
                set_buffer_graph.graph_signature.inputs_to_buffers[node.args[1].name] == "buffer1"):
                set_buffer_self_copy_found = True
    
    # Should have 1 copy operations:
    # 1. The actual copy from input to the narrowed buffer
    assert set_buffer_copy_count == 1, "Expected one copy operation in set_buffer_dynamic method"
    assert set_buffer_self_copy_found is False, "self-copy operation found for buffer1 in set_buffer_dynamic method"
    
    # Check load_from_buffer_dynamic method
    load_buffer_graph = method_graphs["load_from_buffer_dynamic"]
    load_buffer_copy_count = 0
    load_buffer_self_copy_found = False
    
    for node in load_buffer_graph.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.copy_.default:
            load_buffer_copy_count += 1
            # Check if this is a buffer self-copy
            if (node.args[0].name in load_buffer_graph.graph_signature.inputs_to_buffers and 
                node.args[1].name in load_buffer_graph.graph_signature.inputs_to_buffers and
                load_buffer_graph.graph_signature.inputs_to_buffers[node.args[0].name] == "buffer1" and
                load_buffer_graph.graph_signature.inputs_to_buffers[node.args[1].name] == "buffer1"):
                load_buffer_self_copy_found = True
    
    # Should have 2 copy operations:
    # 1. The self-copy for buffer to make sure it's a mutable buffer
    # 2. The actual copy from narrowed buffer to output
    assert load_buffer_copy_count == 2, "Expected two copy operations in load_from_buffer_dynamic method"
    assert load_buffer_self_copy_found, "No self-copy operation found for buffer1 in load_from_buffer_dynamic method"


def test_to_edge(exporter: MultiEntryPointExporter, model: SimpleModel):
    """Test converting to edge format with multiple methods"""
    exporter.register(model.method1, x=MethodArg(torch.ones(10, 20)))
    exporter.register(model.method2, x=MethodArg(torch.ones(10, 20)))
    exporter.export()
    edge_program = exporter.to_edge()
    assert edge_program is not None
    # Verify both methods are present in the edge program
    assert "method1" in edge_program._edge_programs
    assert "method2" in edge_program._edge_programs

def test_to_edge_with_multiple_shared_buffers(exporter: MultiEntryPointExporter, model: SimpleModel):
    """Test converting to edge format with multiple methods and shared buffers"""
    # Register multiple shared buffers
    exporter.register_shared_buffer("buffer1")
    exporter.register_shared_buffer("buffer2")
    exporter.register_shared_buffer("submodule.submodule_buffer1")
    
    # Register multiple methods that use these buffers
    exporter.register(model.method1, x=MethodArg(torch.ones(10, 20)))
    exporter.register(model.method2, x=MethodArg(torch.ones(10, 20)))
    exporter.register(model.get_buffer1)
    exporter.register(model.set_buffer1, x=MethodArg(torch.ones(10, 20)))
    
    # Export and convert to edge
    exporter.export()
    edge_program = exporter.to_edge()
    
    # Verify edge program was created
    assert edge_program is not None
    
    # Verify all methods are present in the edge program
    edge_methods = edge_program._edge_programs
    assert "method1" in edge_methods
    assert "method2" in edge_methods
    assert "get_buffer1" in edge_methods
    assert "set_buffer1" in edge_methods
    assert "et_module_init" in edge_methods  # Verify initialization method is present



def test_to_executorch(exporter: MultiEntryPointExporter, model: SimpleModel):
    """Test converting to executorch format"""
    exporter.register(model.method1, x=MethodArg(torch.ones(10, 20)))
    exporter.export()
    exporter.to_edge()
    executorch_program = exporter.to_executorch()
    assert executorch_program is not None


def test_save(exporter: MultiEntryPointExporter, model: SimpleModel, tmp_path: Path):
    """Test saving the exported model"""
    exporter.register(model.method1, x=MethodArg(torch.ones(10, 20)))
    exporter.export()
    exporter.to_edge()
    exporter.to_executorch()

    output_dir = tmp_path / "test_artifacts"
    output_dir.mkdir(parents=True)

    exporter.save(output_dir, "test_model")
    assert (output_dir / "test_model.pte").exists()
    assert (output_dir / "test_model.etrecord").exists()


def test_method_with_multiple_args(exporter: MultiEntryPointExporter, model: SimpleModel):
    """Test exporting a method with multiple arguments"""
    exporter.register(model.method3, 
                     x=MethodArg(torch.ones(5, 5)),
                     y=MethodArg(torch.ones(5, 5)))
    method_graphs = exporter.export()
    assert "method3" in method_graphs

def test_memory_planning(exporter: MultiEntryPointExporter, model: SimpleModel):
    """Test that memory planning works correctly and is unified between methods"""
    # Register shared buffers
    exporter.register_shared_buffer("other_submodule")
    exporter.register_shared_buffer("buffer1")
    exporter.register_shared_buffer("buffer2")
    exporter.register_shared_buffer("submodule.submodule_buffer1")
    
    # Register methods that use these buffers
    exporter.register(model.method1, x=MethodArg(torch.ones(10, 20)))
    exporter.register(model.method2, x=MethodArg(torch.ones(10, 20)))
    
    # Export and convert to executorch
    exporter.export()
    exporter.to_edge()
    executorch_program = exporter.to_executorch()
    
    # Get memory specs for each method
    memory_specs = {}
    for method_name in executorch_program.methods:
        method_program = executorch_program.exported_program(method_name)
        buffer_specs = {}
        
        # Collect memory specs for buffers in this method
        for node in method_program.graph.nodes:
            if node.op == "placeholder" and node.target in method_program.graph_signature.inputs_to_buffers:
                buffer_name = method_program.graph_signature.inputs_to_buffers[node.target]
                if buffer_name in exporter.registered_shared_buffers:
                    buffer_specs[buffer_name] = {
                        "mem_id": node.meta["spec"].mem_id,
                        "mem_obj_id": node.meta["spec"].mem_obj_id,
                        "mem_offset": node.meta["spec"].mem_offset,
                    }
        memory_specs[method_name] = buffer_specs
    
    # Verify that buffer memory specs are consistent across methods
    # et_module_init should contain all shared buffers, use as a reference.
    init_method = "et_module_init"
    init_specs = memory_specs[init_method]
    
    # Verify all registered shared buffers are present in et_module_init
    for buffer_name in exporter.registered_shared_buffers:
        assert buffer_name in init_specs, f"Shared buffer {buffer_name} not found in et_module_init"
    
    # Verify all buffers in et_module_init are registered shared buffers
    for buffer_name in init_specs:
        assert buffer_name in exporter.registered_shared_buffers, f"Buffer {buffer_name} in et_module_init is not a registered shared buffer"
    
    # For each method, if it has a shared buffer, verify it matches et_module_init
    for method_name, method_specs in memory_specs.items():
        if method_name == init_method:
            continue
        
        for buffer_name, buffer_spec in method_specs.items():
            # If a method has a shared buffer, it must match the init_specs
            assert buffer_name in init_specs, f"Buffer {buffer_name} found in {method_name} but not in et_module_init"
            init_buffer_spec = init_specs[buffer_name]
            
            assert buffer_spec["mem_id"] == init_buffer_spec["mem_id"], \
                f"Buffer {buffer_name} has different mem_id in {method_name}"
            assert buffer_spec["mem_obj_id"] == init_buffer_spec["mem_obj_id"], \
                f"Buffer {buffer_name} has different mem_obj_id in {method_name}"
            assert buffer_spec["mem_offset"] == init_buffer_spec["mem_offset"], \
                f"Buffer {buffer_name} has different mem_offset in {method_name}"
    
    # Verify that buffers have different memory locations (using init_specs as reference)
    buffer_offsets = set()
    for buffer_name, specs in init_specs.items():
        offset = specs["mem_offset"]
        assert offset not in buffer_offsets, f"Buffer {buffer_name} has same offset as another buffer"
        buffer_offsets.add(offset)



def test_dynamic_buffer_set(exporter: MultiEntryPointExporter, model: SimpleModel):
    """Test that dynamic buffer operations work correctly"""
    # Register buffer1 as a shared buffer
    exporter.register_shared_buffer("buffer1")
    
    # Register methods with dynamic buffer operations
    test_dim_0 = Dim("dim0", min=1, max=10)
    test_dim_1 = Dim("dim1", min=1, max=20)
    
    exporter.register(
        model.set_buffer_dynamic, 
        x=MethodArg(torch.ones(5, 20), dynamic_dims={0: test_dim_0, 1: test_dim_1})
    )
    
    method_graphs = exporter.export()
    
    # Check that the method is exported
    assert "set_buffer_dynamic" in method_graphs
    assert "et_module_init" in method_graphs
    
    # First initialize the entire buffer with 3.0
    init_input = torch.full((10, 20), 3.0)
    method_graphs['set_buffer_dynamic'].module()(x=init_input)
    
    # Create test input tensor with specific values
    test_input = torch.full((7, 15), 42.0)  # Different size than example, filled with 42
    
    # Execute the method with smaller region
    method_graphs['set_buffer_dynamic'].module()(x=test_input)
    
    # Get the buffer and verify its contents
    named_buffers = dict(method_graphs['set_buffer_dynamic'].named_buffers())
    assert "buffer1" in named_buffers
    
    # Check that the dynamic region was updated correctly
    buffer_content = named_buffers["buffer1"]
    assert torch.all(buffer_content[:7, :15] == 42.0), "Dynamic region not properly updated"
    assert torch.all(buffer_content[7:, :] == 3.0), "Unchanged region was modified"
    assert torch.all(buffer_content[:7, 15:] == 3.0), "Unchanged region was modified"

def test_dynamic_buffer_load(exporter: MultiEntryPointExporter, model: SimpleModel):
    """Test that dynamic buffer load operations work correctly"""
    # Register buffer1 as a shared buffer
    exporter.register_shared_buffer("buffer1")
    
    # Register methods with dynamic buffer operations
    test_dim_0 = Dim("dim0", min=1, max=10)
    test_dim_1 = Dim("dim1", min=1, max=20)
    
    # Register load method
    exporter.register(
        model.load_from_buffer_dynamic,
        x=MethodArg(torch.ones(5, 20), dynamic_dims={0: test_dim_0, 1: test_dim_1})
    )
    
    method_graphs = exporter.export()
    
    # Check that method is exported
    assert "load_from_buffer_dynamic" in method_graphs
    assert "et_module_init" in method_graphs
    
    # Initialize buffer with known pattern after export using named_buffers
    named_buffers = dict(method_graphs['load_from_buffer_dynamic'].named_buffers())
    assert "buffer1" in named_buffers
    named_buffers["buffer1"].fill_(3.0)
    
    # Create output tensor to load into
    output = torch.zeros(7, 15)  # Different size than example
    
    # Load from buffer into output tensor
    method_graphs['load_from_buffer_dynamic'].module()(x=output)
    
    # Verify the loaded values match what we set
    assert torch.all(output == 3.0), "Loaded values don't match what was in buffer"


if __name__ == "__main__":
    my_model = SimpleModel()
    my_exporter = MultiEntryPointExporter(my_model)
    # test_dynamic_buffer_operations(my_exporter, my_model)
    pass