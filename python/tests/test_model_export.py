from pathlib import Path
import torch
from torch.export import Dim
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from execu_tools.model_exporter import Exporter
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

    def set_cache(self, data: torch.Tensor):
        # self.cache[:,:] = data # this works, but we have 3 caches???
        # get the shape of the date:
        # data_shape = data.shape
        # Dynamically slice based on the target shape
        # slices = tuple(slice(0, dim) for dim in data_shape)
        # batch_sliced = self.cache.narrow(0, 0, data_shape[0])
        # seq_sliced = batch_sliced.narrow(1, 0, data_shape[1])
        self.cache.copy_(data)
        return None

    def get_cache(self, data: torch.Tensor):
        # Note this is fragile b/c of the data dependent-ish slicing.
        # we *must* use narrow, rathar than indexing. There may be otherways to
        # do this, but this did not have to futz with torch._checks.

        data.copy_(self.cache)


def get_test_dir() -> Path:
    return Path(__file__).parent


def test_stateful_export():
    max_batch_size = 10
    max_seq_len = 20

    model = StatefulModel(max_batch_size=max_batch_size, max_seq_len=max_seq_len)

    model.set_cache(torch.ones(max_batch_size, max_seq_len))

    tensor = torch.ones(max_batch_size, max_seq_len)+20

    print(model.get_cache(tensor))


    exporter = Exporter(model)

    # register the buffer:
    exporter.register_shared_buffer("cache")

    # register the methods:
    batch_size = Dim("batch_size_dim", min=1, max=max_batch_size)
    seq_len = Dim("seq_len_dim", min=1, max=max_seq_len)

    # exporter.register(
    #     model.set_cache,
    #     # data=(torch.ones(2, 2),{0: batch_size, 1: seq_len}),
    #     data=(torch.ones(max_batch_size, max_seq_len)),
    # )
    exporter.register(
        model.get_cache,
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


def test_stateful_export_load():
    output_dir = get_test_dir() / "export_artifacts"
    et_runtime: Runtime = Runtime.get()
    program: Program = et_runtime.load_program(
        Path(output_dir / "stateful_model.pte"),
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
