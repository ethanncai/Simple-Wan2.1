# framewise_cross_attention.py

from typing import Optional
import torch
import torch.nn as nn
from .attention import attention  # 直接调用你的 attention 实现

class FrameWiseCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        tokens_per_frame: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        dropout: float = 0.0,
        softmax_scale: Optional[float] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.tokens_per_frame = tokens_per_frame
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim, bias=proj_bias)
        self.dropout = dropout
        self.softmax_scale = softmax_scale if softmax_scale is not None else (self.head_dim ** -0.5)
        self.dtype = dtype

    def forward(
        self,
        x: torch.Tensor,  # [B, N, dim], N = T * tokens_per_frame
        c: torch.Tensor,  # [B, T, dim]
    ) -> torch.Tensor:
        B, N, dim = x.shape
        T = c.shape[1]
        assert N == T * self.tokens_per_frame

        # Project
        q = self.q_proj(x)   # [B, N, dim]
        k = self.k_proj(c)   # [B, T, dim]
        v = self.v_proj(c)   # [B, T, dim]

        # Reshape to per-frame: [B*T, Lq, dim] and [B*T, Lk=1, dim]
        q = q.view(B, T, self.tokens_per_frame, dim).view(B * T, self.tokens_per_frame, dim)
        k = k.view(B * T, 1, dim)
        v = v.view(B * T, 1, dim)

        # Multi-head: [B*T, L, num_heads, head_dim]
        q = q.view(B * T, self.tokens_per_frame, self.num_heads, self.head_dim)
        k = k.view(B * T, 1, self.num_heads, self.head_dim)
        v = v.view(B * T, 1, self.num_heads, self.head_dim)

        # Call attention: each frame is an independent batch item
        out = attention(
            q=q,
            k=k,
            v=v,
            dropout_p=self.dropout,
            softmax_scale=self.softmax_scale,
            dtype=self.dtype,
        )  # [B*T, tokens_per_frame, num_heads, head_dim]

        # Reshape back to [B, N, dim]
        out = out.view(B, T, self.tokens_per_frame, self.dim).view(B, N, self.dim)
        out = self.out_proj(out)
        return out

        
import numpy as np
def test_frame_level_precision():
    B, T, tokens_per_frame, dim = 1, 64, 8, 128
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
        out = model(x, c)  # [B, N, dim]

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
