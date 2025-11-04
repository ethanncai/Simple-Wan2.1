# framewise_cross_attention.py

from typing import Optional
import torch
import torch.nn as nn
from .attention import attention
from .model import FrameWiseCrossAttention
import numpy as np
def test_frame_level_precision():
    B, T, tokens_per_frame, dim = 1, 8, 8, 128
    N = T * tokens_per_frame
    target_frame = 7

    # 使用 float32 避免 numpy 不支持 bfloat16 的问题
    dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.zeros(B, N, dim, device=device, dtype=dtype)
    c = torch.zeros(B, T, dim, device=device, dtype=dtype)
    c[:, target_frame, :] = 1.0

    torch.manual_seed(42)
    model = FrameWiseCrossAttention(
        dim=dim,
        num_heads=8,
        tokens_per_frame=tokens_per_frame,
        qkv_bias=True,   # 禁用 bias，保证严格零
        dtype=dtype,
    ).to(device=device, dtype=dtype)

    with torch.no_grad():
        print(f"x.shape:{x.shape}, c.shape:{c.shape}")
        out = model(x, c)  # [B, N, dim]
        print(f"out.shape:{out.shape}")
        # 安全转为 numpy（避免 bfloat16 问题）
        out_np = out.cpu().float().numpy()  # shape: [1, N, dim]

        # Reshape to [T, tokens_per_frame, dim] for per-frame analysis
        out_per_frame = out_np.reshape(T, tokens_per_frame, dim)

        # 打印每帧的统计信息
        print(f"{'Frame':>5} | {'Max(|out|)':>12} | {'L2 Norm':>12} | {'Mean(|out|)':>12}")
        print("-" * 50)
        for t in range(T):
            frame_out = out_per_frame[t]  # [tokens_per_frame, dim]
            abs_vals = np.abs(frame_out)
            max_val = abs_vals.max()
            l2_norm = np.linalg.norm(frame_out)  # Frobenius norm over [8, 128]
            mean_abs = abs_vals.mean()
            print(f"{t:5d} | {max_val:12.6f} | {l2_norm:12.6f} | {mean_abs:12.6f}")

if __name__ == "__main__":
    test_frame_level_precision()
