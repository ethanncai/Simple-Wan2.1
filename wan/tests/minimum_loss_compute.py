# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import math
import os
import torch
import torch.nn.functional as F
import torch.cuda.amp as amp
from decord import VideoReader, cpu
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image

from ..modules.model import WanModel
from ..modules.t5 import T5EncoderModel
from ..modules.vae import WanVAE

def load_and_preprocess_video(video_path, frame_num=81, target_size=(480, 832)):
    """
    Returns:
        torch.Tensor: shape (C, T, H, W), dtype float32, range [-1, 1]
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)

    if total_frames == 0:
        raise ValueError(f"Video {video_path} has 0 frames.")

    # 确定实际要读取的原始帧索引
    if total_frames >= frame_num:
        indices = list(range(frame_num))  # [0, 1, 2, ..., frame_num-1]

    frames = vr.get_batch(indices).asnumpy()  # (F, H_orig, W_orig, C), F == frame_num

    H_target, W_target = target_size
    transform = Compose([
        ToTensor(),  # (H, W, C) -> (C, H, W), [0,1]
        Resize((H_target, W_target), antialias=True),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # -> [-1, 1]
    ])

    frame_list = []
    for i in range(frames.shape[0]):
        img = Image.fromarray(frames[i])
        tensor = transform(img)  # (3, H, W)
        frame_list.append(tensor)

    video_tensor = torch.stack(frame_list, dim=1)  # (C, frame_num, H, W)
    return video_tensor

# ======================
# 配置参数
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_dir = "/home/rapverse/workspace_junzhi/Wan2.1/Wan2.1-T2V-1.3B"

t5_checkpoint = os.path.join(checkpoint_dir, 'models_t5_umt5-xxl-enc-bf16.pth')
t5_tokenizer = 'google/umt5-xxl'
vae_checkpoint = os.path.join(checkpoint_dir, 'Wan2.1_VAE.pth')

vae_stride = (4, 8, 8)
patch_size = (1, 2, 2)
text_len = 512
t5_dtype = torch.bfloat16
param_dtype = torch.bfloat16

# 你的要求
size_str = "832*480"  # 注意：W*H
frame_num = 81
W, H = map(int, size_str.split('*'))  # W=832, H=480

# ======================
# 用户输入：视频路径
# ======================
video_path = "/home/rapverse/workspace_junzhi/Wan2.1/t2v_832x480_cat_dancing_like_a_human_20251021_152920.mp4"  # ←←← 请修改为你自己的视频路径！

# ======================
# 加载并预处理视频
# ======================
print(f"Loading video from: {video_path}")
video = load_and_preprocess_video(
    video_path,
    frame_num=frame_num,
    target_size=(H, W)  # 注意：函数期望 (H, W)
)
print(f"Loaded video tensor shape: {video.shape}")  # (3, F, H, W)

# ======================
# 加载模型
# ======================
print("Loading DiT...")
# model = WanModel.from_pretrained(checkpoint_dir)
model = WanModel()
model.eval().requires_grad_(False)
model.to(device)

print("Loading VAE...")
vae = WanVAE(
    vae_pth=vae_checkpoint,
    dtype=param_dtype,
    device=device
)

print("Loading T5...")
text_encoder = T5EncoderModel(
    text_len=text_len,
    dtype=t5_dtype,
    device=torch.device('cpu'),
    checkpoint_path=t5_checkpoint,
    tokenizer_path=t5_tokenizer,
    shard_fn=None
)

# ======================
# 计算 latent shape（用于 seq_len）
# ======================
target_shape = (
    vae.model.z_dim,
    (frame_num - 1) // vae_stride[0] + 1,  # T_z
    H // vae_stride[1],                    # H_z
    W // vae_stride[2]                     # W_z
)
seq_len = math.ceil(
    (target_shape[2] * target_shape[3]) /
    (patch_size[1] * patch_size[2]) *
    target_shape[1]
)
print(f"Latent shape: {target_shape}, seq_len: {seq_len}")

# ======================
# VAE 编码
# ======================
print("Encoding video with VAE...")
video = video.to(device)
with torch.no_grad():
    latents = vae.encode([video])  # 传入 list
x0 = latents[0]  # (C_z, T_z, H_z, W_z)
print(f"x0 latent shape: {x0.shape}")

# ======================
# T5 编码
# ======================
prompt = "cat dancing like a human"
print("Encoding prompt...")
text_encoder.model.to(device)
context = text_encoder([prompt], device)
text_encoder.model.cpu()
context = [t.to(device) for t in context]

# ======================
# Flow Matching 训练目标
# ======================
B = 1
# t = torch.rand(B, device=device)
t = torch.tensor([0.1], device=device)
noise = torch.randn_like(x0)

t_view = t.view(B, 1, 1, 1, 1)
x_t = (1 - t_view) * x0.unsqueeze(0) + t_view * noise.unsqueeze(0)
x_t = x_t.squeeze(0)

target_velocity = noise - x0

# ======================
# 模型前向 & Loss
# ======================
print("Running model forward...")
with amp.autocast(dtype=param_dtype):
    velocity_pred = model(
        x_t.unsqueeze(0),
        t=t,
        context=context,
        seq_len=seq_len
    )[0]

loss = F.mse_loss(velocity_pred, target_velocity)
print(f"\nFlow Matching Loss: {loss.item():.6f}")