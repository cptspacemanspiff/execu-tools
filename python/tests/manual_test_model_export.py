from pathlib import Path
import torch
from torch.export import Dim
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from execu_tools.model_exporter import MultiEntryPointExporter
from executorch.runtime import Runtime, Verification, Program, Method


class subobject(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "subcache",
            torch.ones(10, 20, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "subcache_non_persistent",
            torch.ones(10, 20, dtype=torch.float32),
            persistent=False,
        )
        self.register_parameter(
            "subcache_parameter",
            torch.nn.Parameter(torch.ones(10, 20, dtype=torch.float32)),
        )


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
            "othercache",
            torch.ones(10, 20, dtype=torch.float32),
            persistent=True,
        )
        self.subobject1 = subobject()

    def set_cache(self, data: torch.Tensor):
        self.cache.copy_(data)
        return None
    
    def set_cache_zero(self):
        self.subobject1.subcache.copy_(torch.zeros(10,20))
        return None

    def get_cache(self, data: torch.Tensor):
        # self.cache.copy_(self.cache)
        data.copy_(self.cache)
        # it is also a hack to fix the issue that constants are handled differently
        # than mutable buffers, so we need to mutate the buffer. (TODO: fix this)
        return None

    def get_cache_mutation(self, data: torch.Tensor):
        self.cache.copy_(self.cache)
        data.copy_(self.cache)
        return None

def get_test_dir() -> Path:
    return Path(__file__).parent


def test_stateful_export():
    max_batch_size = 10
    max_seq_len = 20

    model = StatefulModel(max_batch_size=max_batch_size, max_seq_len=max_seq_len)

    exporter = MultiEntryPointExporter(model)

    # register the buffer by fqn ie "cache or subobject.subcache":
    exporter.register_shared_buffer("cache")

    # register the buffer by object (registers all persistant buffers in the object):
    # exporter.register_shared_buffer('othercache')
    # exporter.register_shared_buffer('subobject1')

    # register the methods:
    batch_size = Dim("batch_size_dim", min=1, max=max_batch_size)
    seq_len = Dim("seq_len_dim", min=1, max=max_seq_len)

    exporter.register(
        model.set_cache,
        # data=(torch.ones(2, 2),{0: batch_size, 1: seq_len}),
        data=(torch.ones(max_batch_size, max_seq_len)),
    )
    exporter.register(
        model.get_cache,
        # data=(torch.ones(2, 2), {0: batch_size, 1: seq_len}),
        data=(torch.ones(max_batch_size, max_seq_len)),
    )
    exporter.register(
        model.get_cache_mutation,
        # data=(torch.ones(2, 2), {0: batch_size, 1: seq_len}),
        data=(torch.ones(max_batch_size, max_seq_len)),
    )
    # quantize model:
    # exporter.quantize()
    # export to aten:
    exporter.export()
    # lower model:
    exporter.to_edge()
    # export to executorch:
    exporter.to_executorch()
    # save model:
    output_dir = get_test_dir() / "export_artifacts"

    exporter.save(output_dir, "stateful_model")


# from pathlib import Path

# import torch


# et_runtime: Runtime = Runtime.get()
# program: Program = et_runtime.load_program(
#     Path("/tmp/program.pte"),
#     verification=Verification.Minimal,
# )
# print("Program methods:", program.method_names)
# forward: Method = program.load_method("forward")

# inputs = (torch.ones(2, 2), torch.ones(2, 2))
# outputs = forward.execute(inputs)
# print(f"Ran forward({inputs})")
# print(f"  outputs: {outputs}")


# def test_stateful_export_load():
#     output_dir = get_test_dir() / "export_artifacts"
#     et_runtime: Runtime = Runtime.get()
#     program: Program = et_runtime.load_program(
#         Path(output_dir / "stateful_model.pte"),
#         verification=Verification.Minimal,
#     )
#     print("Program methods:", program.method_names)
#     set_cache = program.load_method("set_cache")
#     print("set_cache loaded")
#     get_cache = program.load_method("get_cache")
#     print("get_cache loaded")


if __name__ == "__main__":
    test_stateful_export()
    # test_stateful_export_load()
    pass