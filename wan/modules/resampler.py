import torch
import torch.nn as nn

class ActionResampler(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1, downsample_ratio=4):
        super().__init__()
        self.dim = dim
        self.downsample_ratio = downsample_ratio
        self.num_latents = None  # 动态计算

        # 注意力层 (Cross-Attention)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        # 可学习的 query（latent tokens）
        self.register_parameter("latent_query", None)

        # 归一化 + 前馈网络
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    def forward(self, x):
        B, T, D = x.shape
        T_out = T // self.downsample_ratio + 1

        # 若第一次调用，初始化 learnable query
        if self.latent_query is None:
            latent_query = torch.randn(T_out, D, device=x.device)
            self.latent_query = nn.Parameter(latent_query)

        # Expand query for each batch
        query = self.latent_query.unsqueeze(0).expand(B, -1, -1)  # (B, T', D)

        # Cross-Attention: query attends to all x
        attn_out, _ = self.attn(
            query=self.norm1(query),
            key=self.norm1(x),
            value=self.norm1(x),
            need_weights=False
        )

        # 残差连接 + FFN
        y = query + attn_out
        y = y + self.ffn(self.norm2(y))

        return y  # (B, T', D)

if __name__ == "__main__":
    B, T, D = 2, 1, 128
    x = torch.randn(B, T, D)
    model = ActionResampler(dim=D, downsample_ratio=4)
    print(x.shape)
    y = model(x)
    print(y.shape)  # -> torch.Size([2, 9, 128])
