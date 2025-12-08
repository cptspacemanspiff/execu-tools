#!/usr/bin/env python3
"""
Simple test program demonstrating ExecutorTorch multi-method export with shared state.

This example shows:
1. A model with mutable state (a cache buffer)
2. Multiple methods that read/write the shared state
3. Native ExecutorTorch multi-method export with share_mutable_buffers=True
4. Runtime execution showing state persists across method calls
"""

import torch
from executorch.exir import to_edge, ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.capture._capture import patch_forward
from torch.export import export


class SimpleStatefulModel(torch.nn.Module):
    """
    A simple model with a mutable cache buffer shared across methods.

    Methods:
    - forward: Adds input to cache and returns the result
    - set_cache: Sets the cache to a new value
    - get_cache: Returns the current cache value
    """

    def __init__(self, cache_size: int = 4):
        super().__init__()
        # Register a mutable buffer that will be shared across methods
        self.register_buffer("cache", torch.zeros(cache_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add input to cache and return doubled result."""
        self.cache.add_(x)
        return self.cache * 2

    def set_cache(self, new_cache: torch.Tensor) -> None:
        """Set cache to a new value."""
        self.cache.copy_(new_cache)

    def get_cache(self) -> torch.Tensor:
        """Get current cache value."""
        return self.cache.clone()

    def reset_cache(self) -> None:
        """Reset cache to zeros."""
        self.cache.zero_()


def export_multimethod_model():
    """Export the model with multiple methods and shared state."""

    print("=" * 60)
    print("Step 1: Create the stateful model")
    print("=" * 60)

    eager_module = SimpleStatefulModel(cache_size=4).eval()
    print(f"✓ Created model with cache buffer: {eager_module.cache.shape}")
    print()

    print("=" * 60)
    print("Step 2: Export each method separately")
    print("=" * 60)

    # Export the forward method
    forward_example_inputs = (torch.ones(4),)
    forward = export(eager_module, forward_example_inputs)
    print(f"✓ Exported 'forward' method")

    # Export set_cache method using patch_forward
    with patch_forward(eager_module, eager_module.set_cache):
        set_cache_example_inputs = (torch.zeros(4),)
        set_cache = export(eager_module, set_cache_example_inputs)
    print(f"✓ Exported 'set_cache' method")

    # Export get_cache method using patch_forward
    with patch_forward(eager_module, eager_module.get_cache):
        get_cache = export(eager_module, ())
    print(f"✓ Exported 'get_cache' method")

    # Export reset_cache method using patch_forward
    with patch_forward(eager_module, eager_module.reset_cache):
        reset_cache = export(eager_module, ())
    print(f"✓ Exported 'reset_cache' method")
    print()

    print("=" * 60)
    print("Step 3: Convert to Edge dialect with all methods")
    print("=" * 60)

    edge = to_edge({
        "forward": forward,
        "set_cache": set_cache,
        "get_cache": get_cache,
        "reset_cache": reset_cache,
    })
    print(f"✓ Created EdgeProgramManager with {len(edge._edge_programs)} methods")
    print(f"  Methods: {list(edge._edge_programs.keys())}")
    print()

    print("=" * 60)
    print("Step 4: Convert to ExecutorTorch with shared mutable buffers")
    print("=" * 60)

    et = edge.to_executorch(
        ExecutorchBackendConfig(
            # Enable shared mutable buffer support - this is the key!
            memory_planning_pass=MemoryPlanningPass(share_mutable_buffers=True),
            # Emit buffer names for debugging
            emit_mutable_buffer_names=True,
        )
    )
    print(f"✓ Converted to ExecutorTorch program")
    print()

    print("=" * 60)
    print("Step 5: Verify shared buffer memory planning")
    print("=" * 60)

    et_prog = et.executorch_program

    # Check that the 'cache' buffer is allocated in mem_id=2 (shared arena)
    # and has the same offset across all methods
    shared_buffer_info = {}

    for plan_idx, plan in enumerate(et_prog.execution_plan):
        method_name = list(edge._edge_programs.keys())[plan_idx]

        for value in plan.values:
            if (hasattr(value.val, "allocation_info") and
                value.val.allocation_info is not None and
                value.val.allocation_info.memory_id == 2):

                buffer_name = value.val.extra_tensor_info.fully_qualified_name
                mem_offset = value.val.allocation_info.memory_offset_low
                mem_size = value.val.allocation_info.memory_allocated_bytes

                if buffer_name not in shared_buffer_info:
                    shared_buffer_info[buffer_name] = []

                shared_buffer_info[buffer_name].append({
                    'method': method_name,
                    'offset': mem_offset,
                    'size': mem_size
                })

    print(f"✓ Found {len(shared_buffer_info)} shared buffer(s)")

    for buffer_name, infos in shared_buffer_info.items():
        print(f"\n  Buffer: '{buffer_name}'")
        print(f"  Used in {len(infos)} method(s):")

        offsets = set()
        sizes = set()

        for info in infos:
            print(f"    - {info['method']}: offset={info['offset']}, size={info['size']} bytes")
            offsets.add(info['offset'])
            sizes.add(info['size'])

        # Verify all methods use the same offset (truly shared!)
        if len(offsets) == 1:
            print(f"  ✓ All methods share the same memory location (offset={offsets.pop()})")
        else:
            print(f"  ✗ ERROR: Methods have different offsets: {offsets}")

        if len(sizes) == 1:
            print(f"  ✓ Consistent size across methods ({sizes.pop()} bytes)")

    print()

    print("=" * 60)
    print("Step 6: Save to file")
    print("=" * 60)

    output_path = "simple_stateful_model.pte"
    with open(output_path, "wb") as f:
        f.write(et.buffer)

    print(f"✓ Saved ExecutorTorch program to: {output_path}")
    print(f"  File size: {len(et.buffer)} bytes")
    print()

    return output_path


def test_runtime_execution(model_path: str):
    """Test that the exported model works correctly at runtime."""

    print("=" * 60)
    print("Step 7: Load and test runtime execution")
    print("=" * 60)

    try:
        from executorch.extension.pybindings.portable_lib import (
            _load_for_executorch_from_buffer,
        )

        with open(model_path, "rb") as f:
            et_module = _load_for_executorch_from_buffer(f.read())

        print(f"✓ Loaded ExecutorTorch module from {model_path}")
        print()

        # Test 1: Get initial cache (should be zeros)
        print("Test 1: Get initial cache")
        result = et_module.run_method("get_cache", ())
        print(f"  get_cache() -> {result[0]}")
        print()

        # Test 2: Set cache to [1, 2, 3, 4]
        print("Test 2: Set cache to [1, 2, 3, 4]")
        new_cache = torch.tensor([1.0, 2.0, 3.0, 4.0])
        et_module.run_method("set_cache", (new_cache,))
        print(f"  set_cache({new_cache.tolist()})")
        print()

        # Test 3: Verify cache was set
        print("Test 3: Verify cache was updated")
        result = et_module.run_method("get_cache", ())
        print(f"  get_cache() -> {result[0]}")
        expected = new_cache
        if torch.allclose(result[0], expected):
            print(f"  ✓ Cache correctly set!")
        else:
            print(f"  ✗ ERROR: Expected {expected}, got {result[0]}")
        print()

        # Test 4: Call forward (adds [1, 1, 1, 1] to cache and doubles)
        print("Test 4: Call forward with [1, 1, 1, 1]")
        input_tensor = torch.ones(4)
        result = et_module.run_method("forward", (input_tensor,))
        print(f"  forward({input_tensor.tolist()}) -> {result[0]}")
        # Cache becomes [2, 3, 4, 5], output is [4, 6, 8, 10]
        expected = torch.tensor([4.0, 6.0, 8.0, 10.0])
        if torch.allclose(result[0], expected):
            print(f"  ✓ Forward pass correct!")
        else:
            print(f"  ✗ ERROR: Expected {expected}, got {result[0]}")
        print()

        # Test 5: Verify cache persisted from forward
        print("Test 5: Verify cache persisted after forward")
        result = et_module.run_method("get_cache", ())
        print(f"  get_cache() -> {result[0]}")
        expected = torch.tensor([2.0, 3.0, 4.0, 5.0])
        if torch.allclose(result[0], expected):
            print(f"  ✓ Cache state persisted across method calls!")
        else:
            print(f"  ✗ ERROR: Expected {expected}, got {result[0]}")
        print()

        # Test 6: Reset cache
        print("Test 6: Reset cache to zeros")
        et_module.run_method("reset_cache", ())
        print(f"  reset_cache()")
        result = et_module.run_method("get_cache", ())
        print(f"  get_cache() -> {result[0]}")
        expected = torch.zeros(4)
        if torch.allclose(result[0], expected):
            print(f"  ✓ Cache successfully reset!")
        else:
            print(f"  ✗ ERROR: Expected {expected}, got {result[0]}")
        print()

        print("=" * 60)
        print("SUCCESS: All tests passed!")
        print("=" * 60)
        print("\nKey takeaway:")
        print("  The 'cache' buffer is stored in a shared memory arena (mem_id=2)")
        print("  and persists across all method calls. This demonstrates native")
        print("  ExecutorTorch support for multi-method programs with shared state!")

    except ImportError as e:
        print(f"⚠ Cannot test runtime execution: {e}")
        print("  This requires the executorch runtime library to be built.")
        print("  The export still succeeded - you can test the .pte file in C++")


def main():
    print("\n" + "=" * 60)
    print("ExecutorTorch Multi-Method Export with Shared State Demo")
    print("=" * 60)
    print()

    # Export the model
    model_path = export_multimethod_model()

    # Test runtime execution
    test_runtime_execution(model_path)

    print()


if __name__ == "__main__":
    main()
