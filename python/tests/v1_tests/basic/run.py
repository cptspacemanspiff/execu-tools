import torch
from executorch.runtime import Runtime

runtime = Runtime.get()

input_tensor = torch.randn(1, 3, 32, 32)
program = runtime.load_program("model.pte")
method = program.load_method("forward")
outputs = method.execute([input_tensor])

print(f"outputs {outputs}")