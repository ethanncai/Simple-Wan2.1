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
from torch.utils.data import DataLoader

from ..modules.model import WanModel
from ..modules.t5 import T5EncoderModel
from ..modules.vae import WanVAE
from ..utils.train_utils import load_and_preprocess_video, encode_video_and_text,print_gpu_memory
from ..dataset.test_dataset import OneShotVideoDataset

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
size_str = "832*480"
W, H = map(int, size_str.split('*'))  # W=832, H=480
frame_num = 81

video_path = "/home/rapverse/workspace_junzhi/Wan2.1/t2v_832x480_cat_dancing_like_a_human_20251021_152920.mp4"
prompt_text = "a cat dancing like a human"

dataset = OneShotVideoDataset(
    video_path=video_path,
    text=prompt_text,
)

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
    pin_memory=True if torch.cuda.is_available() else False,
    collate_fn=lambda batch: ([v for v, t in batch], [t for v, t in batch])
)

# ======================
# 加载模型
# ======================
print("Loading DiT...")
model = WanModel.from_pretrained(checkpoint_dir)
# model.eval().requires_grad_(False)
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
# 推理 / 训练循环（单 batch 测试）
# ======================
for i, (videos, texts) in enumerate(dataloader):
    print(f"Batch {i}:")
    print(f"  Number of videos: {len(videos)}")
    print(f"  Video shape: {videos[0].shape}")   # (C, T, H, W)
    print(f"  Texts: {texts}")

    # Encode videos to latents and texts to embeddings
    latents, context = encode_video_and_text(
        videos=videos,
        texts=texts,
        vae=vae,
        text_encoder=text_encoder,
        vae_stride=vae_stride,
        patch_size=patch_size,
        device=device,
        param_dtype=param_dtype
    )

    print("Latent shape:", latents[0].shape)   # (C_z, T_z, H_z, W_z)
    print("Context shape:", context[0].shape)  # (L, D)

    # Compute sequence length for transformer
    C_z, T_z, H_z, W_z = latents[0].shape
    seq_len = math.ceil(
        (H_z * W_z) / (patch_size[1] * patch_size[2]) * T_z
    )

    # Stack latents into a batch tensor
    x0_batch = torch.stack(latents, dim=0)  # (B, C_z, T_z, H_z, W_z)
    B = x0_batch.shape[0]

    # Sample random timesteps: one per sample in the batch
    t = torch.rand(B, device=device)  # shape: (B,)

    # Add noise: x_t = (1 - t) * x0 + t * noise
    noise = torch.randn_like(x0_batch)
    t_view = t.view(B, 1, 1, 1, 1)  # now always valid: (B, 1, 1, 1, 1)
    x_t = (1 - t_view) * x0_batch + t_view * noise
    target_velocity = noise - x0_batch  # velocity objective

    # Model forward
    print("Running model forward...")
    print_gpu_memory("before forward with grad")
    with amp.autocast(dtype=param_dtype):
        velocity_pred_list = model(
            x_t,               # (B, C_z, T_z, H_z, W_z)
            t=t,               # (B,)
            context=context,   # List[(L, D)] of length B
            seq_len=seq_len
        )  # output: (B, C_z, T_z, H_z, W_z)
    print_gpu_memory("after forward with grad")
    velocity_pred = torch.stack(velocity_pred_list, dim=0)
    print(velocity_pred.shape)
    loss = F.mse_loss(velocity_pred, target_velocity)
    print(f"\nFlow Matching Loss: {loss.item():.6f}")

    # Only run one batch for testing
    break