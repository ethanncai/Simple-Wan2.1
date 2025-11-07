import torch
import torch.nn as nn
import math

def _sinusoidal_positional_encoding(T, dim, device, dtype=torch.float32, max_period=10000):
    pe = torch.zeros(T, dim, device=device, dtype=dtype)
    position = torch.arange(0, T, dtype=dtype, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=dtype, device=device) *
        (-math.log(float(max_period)) / dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class ActionResampler(nn.Module):
    def __init__(self, action_dim, dim, num_heads=8, dropout=0.1, downsample_ratio=4):
        super().__init__()
        self.action_dim = action_dim
        self.dim = dim
        self.downsample_ratio = downsample_ratio

        self.input_proj = nn.Linear(action_dim, dim)
        self.latent_token = nn.Parameter(torch.randn(1, dim))

        self.attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm_latent = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, T, _ = x.shape
        T_out = max(1, (T + self.downsample_ratio - 1) // self.downsample_ratio)

        x = self.input_proj(x)  # (B, T, dim)
        # 使用与 x 相同的 dtype
        pos_enc = _sinusoidal_positional_encoding(T, self.dim, x.device, dtype=x.dtype)
        x = x + pos_enc.unsqueeze(0)

        base_latent = self.latent_token.expand(B, T_out, -1)
        latent_pos = _sinusoidal_positional_encoding(T_out, self.dim, x.device, dtype=x.dtype).unsqueeze(0)
        query = base_latent + latent_pos

        x_norm = self.norm1(x)
        query_norm = self.norm_latent(query)
        attn_out, _ = self.attn(query=query_norm, key=x_norm, value=x_norm, need_weights=False)

        y = query + attn_out
        y = y + self.ffn(self.norm2(y))
        return y


if __name__ == "__main__":
    B, T, action_dim = 2, 81, 7  # 机械臂 7 维动作
    x = torch.randn(B, T, action_dim)

    model = ActionResampler(
        action_dim=action_dim,
        dim=128,
        downsample_ratio=4
    )

    y = model(x)
    print("Input shape:", x.shape)   # [2, 81, 7]
    print("Output shape:", y.shape)  # [2, 21, 128] (因为 81//4 ≈ 20.25 → 向上取整为 21)