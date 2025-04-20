import torch

if torch.cuda.is_available():
    print("CUDA 可用")
else:
    print("CUDA 不可用")