from pathlib import Path
import torch
from torch.export import Dim
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from execu_tools.model_exporter import Exporter

class StatefulModel(torch.nn.Module):
    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.register_buffer(
            "cache", torch.zeros((max_batch_size, max_seq_len), dtype=torch.float32)
        )

    def set_cache(self, data: torch.Tensor):
        # get the shape of the date:
        data_shape = data.shape
        # Dynamically slice based on the target shape
        slices = tuple(slice(0, dim) for dim in data_shape)
        self.cache[slices] = data

    def get_cache(self, data: torch.Tensor):
        # Note this is fragile b/c of the data dependent-ish slicing. 
        # we *must* use narrow, rathar than indexing. There may be otherways to 
        # do this, but this did not have to futz with torch._checks.
        shape = data.shape
        batch_sliced = self.cache.narrow(0,0, shape[0])
        seq_sliced = batch_sliced.narrow(1,0, shape[1])
        data[:shape[0], :shape[1]] = seq_sliced

def get_test_dir() -> Path:
    return Path(__file__).parent

def test_stateful_export():
    max_batch_size = 10
    max_seq_len = 20

    model = StatefulModel(max_batch_size=max_batch_size, max_seq_len=max_seq_len)

    exporter = Exporter(model)

    # register the methods:
    batch_size = Dim("batch_size_dim", min=1, max=max_batch_size)
    seq_len = Dim("seq_len_dim", min=1, max=max_seq_len)

    exporter.register(
        model.set_cache,
        data=(torch.ones(3,3), {0: batch_size, 1: seq_len}),
    )
    exporter.register(
        model.get_cache,
        data=(torch.ones(2, 2), {0: batch_size, 1: seq_len}),
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
    output_dir = get_test_dir()/'export_artifacts'

    exporter.save(output_dir,"stateful_model")

if __name__ == "__main__":
    test_stateful_export()