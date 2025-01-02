# Load model directly

from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from execu_tools.encoder_decoder_export import (
    EncoderDecoderExportable,
    EncoderDecoderExportableConfig,
    StatefulModel,
    patch_forward,
)

import torch
import torch.export._trace

from torch.export import export, Dim, ExportedProgram
from executorch.exir import (
    to_edge,
    to_edge_transform_and_lower,
    EdgeCompileConfig,
    EdgeProgramManager,
)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
import inspect


def test_stateful_export():
    max_batch_size = 10
    max_seq_len = 20

    model = StatefulModel(max_batch_size=max_batch_size, max_seq_len=max_seq_len)

    # model.set_cache(torch.ones(2, 7))

    # cache_data = torch.zeros(3, max_seq_len)
    # model.get_cache(cache_data)

    # BaseExportableModel(model)
    class register_helper:
        registered_method_dict: dict[str, tuple[callable, dict]]
        method_graphs: dict[str, ExportedProgram]

        def __init__(self):
            self.registered_method_dict = {}
            self.method_graphs = {}

        def register(self, fn, **kwargs):
            self.registered_method_dict[fn.__name__] = (fn, kwargs)

        def trace_model(self, model):
            with torch.no_grad():
                for method in self.registered_method_dict:
                    fn, kwargs = self.registered_method_dict[method]
                    # update the forward method of the model:
                    with patch_forward(model, fn):
                        sig = inspect.signature(model.forward)
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
                            model,
                            (),
                            kwargs=example_args,
                            dynamic_shapes=dynamic_shapes,
                            # strict=True
                        )
                        self.method_graphs[fn.__name__] = method_graph

        # def quantize_model(self):
        #     # check that we have a model traced (model graphs not empty):
        #     if len(self.method_graphs) == 0:
        #         raise ValueError("No method graphs found. Please trace the model first.")
        #     pass

        def export_model(self, partitioners: list = None) -> EdgeProgramManager:
            # export the model graphs to the edge:
            if partitioners is None:
                executorch_program = to_edge(self.method_graphs)
            else:
                executorch_program: EdgeProgramManager = to_edge_transform_and_lower(
                    self.method_graphs,
                    partitioner=partitioners,
                )

            for method in executorch_program.methods:
                print(f"Edge Dialect graph of {method}")
                print(executorch_program.exported_program(method))

            return executorch_program

        def save_executorch_program(self, path: str):
            

    registry = register_helper()

    batch_size = Dim("batch_size_dim", min=1, max=max_batch_size)
    seq_len = Dim("seq_len_dim", min=1, max=max_seq_len)

    registry.register(
        model.set_cache,
        data=(torch.ones(3,3), {0: batch_size, 1: seq_len}),
    )
    # pass an example: (this removes the case where you have 2 types of inputs, rathar than a dynamic)
    registry.register(
        model.get_cache,
        data=(torch.ones(2, 2), {0: batch_size, 1: seq_len}),
    )

    # debug notes: when we have a batch size, we create a symbolic object of the size s0.
    # that symbolic value is used to index into the cache.

    registry.trace_model(model)
    exported_program = registry.export_model([XnnpackPartitioner()])
    # exported_program = registry.export_model()
    # exported_program.save("encoder_decoder_export.pte")
    print("Successfully exported model")




def test_encoder_decoder_export(model_name="Helsinki-NLP/opus-mt-en-fr"):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name
        # attn_implementation=attn_implementation,
    )

    cache_implementation = "static"
    max_batch_size = 4
    max_generation_length = 200

    model.generation_config.update(
        use_cache=True,
        cache_implementation=cache_implementation,
        max_length=max_generation_length,
        cache_config={
            "batch_size": max_batch_size,
            "max_cache_len": max_generation_length,
        },
    )

    exportable_config = EncoderDecoderExportableConfig(
        min_batch_size=1,
        max_batch_size=max_batch_size,
        min_encoder_seq_len=1,
        max_encoder_seq_len=max_generation_length,
        min_decoder_seq_len=1,
        max_decoder_seq_len=max_generation_length,
        cache_dtype=torch.float32,
    )
    exportable = EncoderDecoderExportable(model, exportable_config)

    print("done")

    with torch.no_grad():
        # TODO: The default inputs only work for text models. We need to add support for vision/audio models.
        example_input_ids = torch.tensor([[10]], dtype=torch.long)
        example_attention_mask = torch.tensor([[1]], dtype=torch.bool)

        # functions to export:
        func_list = ["forward_encoder"]  # , "forward_decoder"]

        programs: dict[str, torch.export.ExportedProgram] = {}

        # patch the forward method:
        for func_name in func_list:
            # function_attr = getattr(exportable, func_list[0])
            with patch_forward(exportable, exportable.forward_2):
                kwargs = {
                    "encoder_input_ids": example_input_ids,
                    "encoder_attention_mask": example_attention_mask,
                }
                # validate that it runs:
                # exportable(**kwargs)

                # Due to issue https://github.com/pytorch/pytorch/issues/128394, we need to switch to use an internal
                # export API and pre_dispatch=False. Switch to use the public API once the issue is included in 2.5 release.
                exported_program = export(
                    exportable,
                    (example_input_ids, example_attention_mask),
                    # pre_dispatch=False,
                    strict=True,
                )
                programs[func_name] = exported_program

        return programs


if __name__ == "__main__":
    # test_encoder_decoder()
    # test_encoder_decoder_export()
    test_stateful_export()



import torch
from torch.export import export, export_for_training, ExportedProgram


# class M(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.param = torch.nn.Parameter(torch.rand(3, 4))
#         self.linear = torch.nn.Linear(4, 5)

#     def forward(self, x):
#         return self.linear(x + self.param).clamp(min=0.0, max=1.0)


# example_args = (torch.randn(3, 4),)
# pre_autograd_aten_dialect = export_for_training(M(), example_args).module()
# # Optionally do quantization:
# # pre_autograd_aten_dialect = convert_pt2e(prepare_pt2e(pre_autograd_aten_dialect, CustomBackendQuantizer))
# aten_dialect: ExportedProgram = export(pre_autograd_aten_dialect, example_args)
# edge_program: exir.EdgeProgramManager = exir.to_edge(aten_dialect)
# # Optionally do delegation:
# # edge_program = edge_program.to_backend(CustomBackendPartitioner)
# executorch_program: exir.ExecutorchProgramManager = edge_program.to_executorch(
#     ExecutorchBackendConfig(
#         passes=[],  # User-defined passes
#     )
# )

# with open("model.pte", "wb") as file:
#     file.write(executorch_program.buffer)

# def test_encoder_decoder(model_name="Helsinki-NLP/opus-mt-en-fr"):
#     """Validate that the model works at all.

#     Args:
#         model_name (str, optional): _description_. Defaults to "Helsinki-NLP/opus-mt-en-fr".
#     """
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSeq2SeqLM.from_pretrained(
#         model_name
#         # attn_implementation=attn_implementation,
#     )

#     cache_implementation = "static"
#     max_batch_size = 4
#     max_generation_length = 200

#     strings_1 = [
#         "When the night has come and the land is dark, and the moon is the only light we will see.",
#         "Abba is the best",
#         # "When the night has come and the land is dark, and the moon is the only light we will see.",
#         # "When the night has come and the land is dark, and the moon is the only light we will see.",
#     ]
#     input_ids = tokenizer(strings_1, return_tensors="pt", padding=True)
#     tokens = model.generate(**input_ids)
#     text_translated = [tokenizer.decode(t, skip_special_tokens=False) for t in tokens]
#     print(text_translated)
