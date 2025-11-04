# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
from typing import Optional
import torch
from torch import amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from ..utils.train_utils import compute_dispersion_moduli
from .attention import flash_attention, attention
from .resampler import ActionResampler
# from .fw_attn import FrameWiseCrossAttention
__all__ = ['WanModel']

T5_CONTEXT_TOKEN_NUMBER = 512
FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER = 257 * 2


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast("cuda", enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast("cuda", enabled=False)
def rope_apply(x, grid_sizes, freqs):
    orig_dtype = x.dtype
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).to(orig_dtype)


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        dtype_in = x.dtype
        compute_dtype = self.weight.dtype
        y = self._norm(x.to(compute_dtype))
        y = y * self.weight  # same dtype for both
        return y.to(dtype_in)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        dtype_in = x.dtype
        # If affine, ensure input matches parameter dtype; otherwise use float32 compute.
        param = getattr(self, 'weight', None)
        compute_dtype = param.dtype if isinstance(param, torch.Tensor) else torch.float32
        y = super().forward(x.to(compute_dtype))
        return y.to(dtype_in)


import torch
import torch.nn as nn

class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        # === Zero-init output projection ===
        nn.init.zeros_(self.o.weight)
        if self.o.bias is not None:
            nn.init.zeros_(self.o.bias)

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)
        v = v.to(q.dtype)

        x = flash_attention(
            q=q,
            k=k,
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size
        )

        x = x.flatten(2)
        x = self.o(x)  # zero-init makes it no-op at start
        return x


class WanCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): [B, L1, C]
            context(Tensor): [B, L2, C]
            context_lens(Tensor): [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        v = v.to(q.dtype)

        x = flash_attention(q, k, v, k_lens=context_lens)
        x = x.flatten(2)
        x = self.o(x)  # zero-init -> short-circuited initially
        return x

class FrameWiseCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size=(-1, -1),
        qk_norm=True,
        tokens_per_frame=None,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        dropout: float = 0.0,
        softmax_scale: Optional[float] = None,
        dtype: torch.dtype = torch.bfloat16,
        use_fw_pos_embed: bool = True,  # 新增开关
        eps=1e-6
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
        self.use_pos_embed = use_fw_pos_embed
        self.norm1 = WanLayerNorm(dim, eps)
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)

        if use_fw_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(tokens_per_frame, dim))
            nn.init.normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

    def forward(
        self,
        x: torch.Tensor,  # [B, N, dim], N = T * tokens_per_frame
        e,
        action: torch.Tensor,  # [B, T, dim]
        seq_lens,
        grid_sizes,
        freqs,
    ) -> torch.Tensor:
        B, N, dim = x.shape
        T = action.shape[1]
        assert N == T * self.tokens_per_frame

        # === 添加位置编码 ===
        if self.use_pos_embed:
            # x: [B, T, tokens_per_frame, dim]
            x = x.view(B, T, self.tokens_per_frame, dim)
            # 广播 pos_embed [tokens_per_frame, dim] -> [B, T, tokens_per_frame, dim]
            x = x + self.pos_embed.unsqueeze(0).unsqueeze(0)
            x = x.view(B, N, dim)

        # Project
        q = self.q_proj(x)   # [B, N, dim]
        k = self.k_proj(action)   # [B, T, dim]
        v = self.v_proj(action)   # [B, T, dim]

        # Reshape to per-frame: [B*T, Lq, dim] and [B*T, Lk=1, dim]
        q = q.view(B, T, self.tokens_per_frame, dim).view(B * T, self.tokens_per_frame, dim)
        k = k.view(B * T, 1, dim)
        v = v.view(B * T, 1, dim)

        # Multi-head: [B*T, L, num_heads, head_dim]
        q = q.view(B * T, self.tokens_per_frame, self.num_heads, self.head_dim)
        k = k.view(B * T, 1, self.num_heads, self.head_dim)
        v = v.view(B * T, 1, self.num_heads, self.head_dim)

        # Call attention
        out = attention(
            q=q,
            k=k,
            v=v,
            dropout_p=self.dropout,
            softmax_scale=self.softmax_scale,
            dtype=self.dtype,
        )  # [B*T, tokens_per_frame, num_heads, head_dim]

        # Reshape back to [B, N, dim]
        x = x.view(B, T, self.tokens_per_frame, self.dim).view(B, N, self.dim)
        x = self.out_proj(x)
        e = e.clone()
        assert e.dtype == torch.float32
        with amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # self-attention
        norm1x = self.norm1(x)
        sa_inp = norm1x * (1 + e[1].to(norm1x.dtype)) + e[0].to(norm1x.dtype)
        y = self.self_attn(sa_inp, seq_lens, grid_sizes, freqs)
        x = x + (y.to(x.dtype) * e[2].to(x.dtype))
        return x

class WanAttentionBlock(nn.Module):

    def __init__(self,
                #  cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim,
                                            num_heads,
                                            (-1, -1),
                                            qk_norm,
                                            eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        
        assert e.dtype == torch.float32
        with amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # self-attention
        norm1x = self.norm1(x)
        sa_inp = norm1x * (1 + e[1].to(norm1x.dtype)) + e[0].to(norm1x.dtype)
        y = self.self_attn(sa_inp, seq_lens, grid_sizes, freqs)
        x = x + (y.to(x.dtype) * e[2].to(x.dtype))

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            y_ca = self.cross_attn(self.norm3(x), context, context_lens)
            x = x + y_ca.to(x.dtype)
            norm2x = self.norm2(x)
            ffn_inp = norm2x * (1 + e[4].to(norm2x.dtype)) + e[3].to(norm2x.dtype)
            y = self.ffn(ffn_inp)
            x = x + (y.to(x.dtype) * e[5].to(x.dtype))
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x

class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        with amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        ex0, ex1 = e[0].to(x.dtype), e[1].to(x.dtype)
        x = self.head(self.norm(x) * (1 + ex1) + ex0)
        return x


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                #  model_type='t2v',
                 target_latent_w = 208,
                 target_latent_h = 120,
                 target_latent_t = 21,
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=1536,
                 ffn_dim=8960,
                 freq_dim=256,
                 text_dim=4096,
                 action_dim=16,
                 out_dim=16,
                 num_heads=12,
                 num_layers=30,
                 num_img_conditioned_layers=5,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video) or 'flf2v' (first-last-frame-to-video) or 'vace'
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        # assert model_type in ['t2v', 'i2v', 'flf2v', 'vace']
        # self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.action_dim = action_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # precalculate the shape related params

        assert patch_size[0] == 1, "we currently only handle time patch stride == 1"
        target_w_patches = target_latent_w // patch_size[2]
        target_h_patches = target_latent_h // patch_size[1]
        self.tokens_per_frame = target_w_patches * target_h_patches

        self.grid_size = torch.tensor([target_latent_t, target_h_patches, target_w_patches])

        # alpha
        self.alpha = nn.Parameter(torch.tensor(1e-3))

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # clip projection
        CLIP_DIM = 768
        VAE_T_RATIO = 4
        self.projector = nn.Linear(CLIP_DIM, dim, bias=True)

        self.action_resampler = ActionResampler(
            action_dim=self.action_dim,
            dim=self.dim,
            downsample_ratio=VAE_T_RATIO) # B,T_pixle,action_dim -> B, T_latent,dim

        # blocks
        self.blocks = nn.ModuleList([
            WanAttentionBlock(dim=dim, ffn_dim=ffn_dim, num_heads=num_heads,
                              window_size=window_size, qk_norm=qk_norm, cross_attn_norm=cross_attn_norm, eps=eps)
            for _ in range(num_layers)
        ])

        self.action_blocks = nn.ModuleList([
            FrameWiseCrossAttention(dim=dim, num_heads=num_heads,
                              window_size=window_size, qk_norm=qk_norm, tokens_per_frame=self.tokens_per_frame,eps=eps)
            for _ in range(num_img_conditioned_layers)
        ])

        self.transformer_block_lists = [self.blocks, self.action_blocks]
        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        max_len = max([len(block) for block in self.transformer_block_lists])
        self.moduli = compute_dispersion_moduli(self.transformer_block_lists)
        assert max_len == len(self.blocks), f"max len is {max_len}, not block len: {len(self.blocks)}"


    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        img_latent=None,
        clip_feat=None,
        action=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            img (List[Tensor]):
                List of input initial frame condition each with shape [C_in, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode or first-last-frame-to-video mode
            action (Tensor, [B, L, action_dim])
        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # if self.model_type == 'i2v' or self.model_type == 'flf2v':
        #     assert clip_fea is not None and y is not None
        # params

        # process action
        assert len(x) == len(img_latent) and len(x) == len(context), "inconsists len"
        B = len(x)
        action = torch.stack(action) # batchify to B, raw_frame, dim
        action_aligned = self.action_resampler(action) # (B,T_pixle,action_dim) -> (B, T_latent,dim)

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # latent space hard replace
        x_ = []
        for u,img in zip(x,img_latent):
            u_new = u.clone()
            u_new[:, 0] = img[:, 0]
            x_.append(u_new)
        x = x_
            
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        clip_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        clip_feat = self.projector(clip_feat) # B, dim
        clip_feat = clip_feat.unsqueeze(1) # B, 1, dim
            # context_clip = self.img_emb(clip_fea)  # bs x 257 (x2) x dim
            # context_iv = torch.concat([context_clip, context], dim=1)

        grid_sizes = self.grid_size.unsqueeze(0).repeat(B,1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes, # [latent_t, h_patches, w_patches]
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            clip_feat=clip_feat, # B,dim
            clip_lens=clip_lens,
            action=action_aligned
            )
        
        # altering_forward = create_alternating_forward(self.blocks, self.i2v_extension_blocks, grad_checkpoint=grad_checkpoint)
        block_moduli, img_moduli = self.moduli
        assert block_moduli == 1 # main block

        for i in range(len(self.blocks)):
            x = grad_checkpoint(
                self.blocks[i],
                x,
                kwargs['e'],
                kwargs['seq_lens'],
                kwargs['grid_sizes'],
                kwargs['freqs'],
                kwargs['context'],
                kwargs['context_lens'],
                use_reentrant=False
            )
            x_ = None
            if i % img_moduli == 0:
                x_ = grad_checkpoint(
                    self.action_blocks[i // img_moduli],
                    x,
                    kwargs['e'],
                    kwargs['action'],
                    kwargs['seq_lens'],
                    kwargs['grid_sizes'],
                    kwargs['freqs'],
                    use_reentrant=False
                )
            x = x if x_ is None else x + self.alpha * x_
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
        
def test_wan_model():
    print("Testing WanModel forward pass with random data...")
    
    # Set parameters for testing (smaller sizes for quick testing)
    batch_size = 2
    in_dim = 16  # Input channels
    out_dim = 16  # Output channels
    dim = 256     # Hidden dimension (reduced for testing)
    ffn_dim = 1024  # Feed-forward network dimension
    num_heads = 4  # Number of attention heads
    num_layers = 30  # Number of transformer layers
    text_dim = 4096  # Text embedding dimension
    frames = 21  # Number of frames
    height = 120  # Frame height
    width = 208   # Frame width
    patch_size = (1, 2, 2)  # Temporal and spatial patch sizes
    text_len = 77  # Maximum text tokens
    freq_dim = 256  # Time embedding dimension
    action_dim = 16
    raw_action_frame = 81
    target_latent_w = width
    target_latent_h = height
    target_latent_t = frames
    
    # Create the model - test with t2v mode
    model = WanModel(
        target_latent_w = target_latent_w,
        target_latent_h = target_latent_h,
        target_latent_t = target_latent_t,
        patch_size=patch_size,
        text_len=text_len,
        in_dim=in_dim,
        dim=dim,
        ffn_dim=ffn_dim,
        freq_dim=freq_dim,
        text_dim=text_dim,
        out_dim=out_dim,
        num_heads=num_heads,
        num_layers=num_layers
    ).to('cuda')
    
    # Set to evaluation mode
    model.eval()
    
    # Generate random input data
    x = [torch.randn(in_dim, frames, height, width).to('cuda') for _ in range(batch_size)]
    action = [torch.randn(raw_action_frame,action_dim).to('cuda') for _ in range(batch_size)]
    img = [torch.randn(in_dim,1, height, width).to('cuda') for _ in range(batch_size)]
    t = torch.randint(0, 1000, (batch_size,)).to('cuda')
    clip_feat = torch.randn([batch_size,768]).to('cuda') 
    context = [torch.randn(text_len, text_dim).to('cuda') for _ in range(batch_size)]
    
    # Calculate sequence length based on patched dimensions
    f_patches = frames // patch_size[0]
    h_patches = height // patch_size[1]
    w_patches = width // patch_size[2]
    seq_len = f_patches * h_patches * w_patches
    
    # Run forward pass
    with torch.no_grad():
        outputs = model(x, t, context, seq_len,img_latent=img,clip_feat=clip_feat,action=action)
    
    # Check output shape
    expected_f = frames
    expected_h = height // patch_size[1]
    expected_w = width // patch_size[2]
    
    print(f"Input shapes: {[i.shape for i in x]}")
    print(f"Output shapes: {[o.shape for o in outputs]}")
    
if __name__ == "__main__":
    test_wan_model()
    
    