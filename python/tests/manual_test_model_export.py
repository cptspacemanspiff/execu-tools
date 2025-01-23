from pathlib import Path
import torch
from torch.export import Dim
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from execu_tools.model_exporter import MultiEntryPointExporter, MethodArg
from executorch.runtime import Runtime, Verification, Program, Method

class StatefulModel(torch.nn.Module):
    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.register_buffer(
            "cache",
            torch.ones((max_batch_size, max_seq_len), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "cache2",
            torch.ones((max_batch_size, max_seq_len*2), dtype=torch.float32),
            persistent=True,
        )

    def set_cache(self, data: torch.Tensor):
        cache_slice_0 = self.cache.narrow(0, 0, data.size(0))
        cache_slice_1 = cache_slice_0.narrow(1, 0, data.size(1))
        cache_slice_1.copy_(data)
        return None
    
    def get_cache(self, data: torch.Tensor):
        cache_slice_0 = self.cache.narrow(0, 0, data.size(0))
        cache_slice_1 = cache_slice_0.narrow(1, 0, data.size(1))
        data.copy_(cache_slice_1)
        return None


def get_test_dir() -> Path:
    return Path(__file__).parent


def test_stateful_export():
    max_batch_size = 10
    max_seq_len = 20

    model = StatefulModel(max_batch_size=max_batch_size, max_seq_len=max_seq_len)
    exporter = MultiEntryPointExporter(model)

    # Register the buffer by fqn
    exporter.register_shared_buffer("cache")
    exporter.register_shared_buffer("cache2")
    # Optional: register other buffers
    # exporter.register_shared_buffer('othercache')
    # exporter.register_shared_buffer('subobject1')

    # Define dynamic dimensions
    batch_size = Dim("batch_size_dim", min=1, max=max_batch_size)
    seq_len = Dim("seq_len_dim", min=1, max=max_seq_len)

    # Register methods with dynamic dimensions
    exporter.register(
        model.set_cache,
        data=MethodArg(
            torch.ones(max_batch_size, max_seq_len),
            dynamic_dims={0: batch_size, 1: seq_len}
        ),
    )
    
    exporter.register(
        model.get_cache,
        data=MethodArg(
            torch.ones(max_batch_size, max_seq_len),
            dynamic_dims={0: batch_size, 1: seq_len}
        ),
    )
    # Export process
    exporter.export()
    exporter.to_edge()
    exporter.to_executorch()

    # Save model
    output_dir = get_test_dir() / "export_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    exporter.save(output_dir, "stateful_model")


def test_stateful_export_load():
    """Test loading and running the exported model"""
    output_dir = get_test_dir() / "export_artifacts"
    et_runtime: Runtime = Runtime.get()
    program: Program = et_runtime.load_program(
        output_dir / "stateful_model.pte",
        verification=Verification.Minimal,
    )
    print("Program methods:", program.method_names)
    
    set_cache = program.load_method("set_cache")
    print("set_cache loaded")
    
    get_cache = program.load_method("get_cache")
    print("get_cache loaded")

    


if __name__ == "__main__":
    test_stateful_export()
    # test_stateful_export_load()