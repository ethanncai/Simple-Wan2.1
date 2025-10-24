import torch
import torch.nn as nn

x = torch.randn(2, 128, dtype=torch.float32, device="cuda")
linear = nn.Linear(128, 64).cuda()

with torch.autocast("cuda", dtype=torch.bfloat16):
    print("Output dtype:", x.dtype)
    y = linear(x)
    print("Output dtype:", y.dtype)  # 应该输出 torch.bfloat16