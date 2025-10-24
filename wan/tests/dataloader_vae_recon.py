import os
import torch
from decord import VideoReader, cpu
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import imageio.v3 as iio
import numpy as np
from torch.utils.data import DataLoader
from ..modules.vae import WanVAE
from ..dataset.test_dataset import OneShotVideoDataset

def tensor_to_uint8_rgb(video_tensor):
    """
    将 [-1,1] 的 (C, F, H, W) tensor 转为 (F, H, W, C) uint8 RGB numpy array
    """
    video_tensor = video_tensor.clamp(-1, 1)
    video_tensor = (video_tensor + 1) / 2  # [-1,1] -> [0,1]
    video_tensor = video_tensor.permute(1, 2, 3, 0)  # (F, H, W, C)
    return (video_tensor * 255).byte().cpu().numpy()


video_path = "/home/rapverse/workspace_junzhi/dataset/train/392-651388-007/head_color.mp4"
prompt_text = ["POV of a pair of junzhi robot arm doing something",
                "POV of junzhi robot and it is doing something",
                "First person view of a junzhi brand robot arm and it is doing something",
                "POV of a pair of junzhi robot gripper",
                "First person view of a junzhi brand robot gripper and it is doing something",]

output_orig_path = "./original_resized.mp4"
output_recon_path = "./reconstructed_video.mp4"

def collate_fn(batch):
    videos, texts = zip(*batch)
    return list(videos), list(texts)
dataset = OneShotVideoDataset(video_path=video_path, text=prompt_text)
dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=16,
    pin_memory=True,
    collate_fn=collate_fn
)

# ======================
# 配置
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_dir = "/home/rapverse/workspace_junzhi/Wan2.1/Wan2.1-T2V-1.3B"
vae_checkpoint = os.path.join(checkpoint_dir, 'Wan2.1_VAE.pth')

# ======================
# 加载 VAE
# ======================
print("Loading WanVAE...")
vae = WanVAE(
    vae_pth=vae_checkpoint,
    dtype=torch.bfloat16,
    device=device
)
# print(vae.scale.dtype)
# ======================
# Encode + Decode
# ======================
for batch_idx, batch in enumerate(dataloader):
    # Unpack (video, text) pairs
    videos, texts = batch
    assert isinstance(videos,list)
    assert isinstance(texts,list)

    video_tensor = videos[0]
    videos = [v.to(device, dtype=torch.bfloat16) for v in videos]
    print("Encoding...")
    with torch.no_grad():
        latents = vae.encode(videos)
        print(f"Latent shape: {latents[0].shape}")

        print("Decoding...")
        recon_list = vae.decode(latents)
    recon_video = recon_list[0]  # (C, F, H, W)

    print(f"Reconstructed shape: {recon_video.shape}")

    # ======================
    # 转为 numpy 并保存
    # ======================
    print("Converting tensors to uint8 RGB...")
    orig_frames = tensor_to_uint8_rgb(video_tensor)
    recon_frames = tensor_to_uint8_rgb(recon_video)

    # 使用 imageio 保存（自动调用 ffmpeg，兼容性强）
    print(f"Saving original (resized) video to {output_orig_path}...")
    iio.imwrite(output_orig_path, orig_frames, fps=30, codec="libx264", quality=10)

    print(f"Saving reconstructed video to {output_recon_path}...")
    iio.imwrite(output_recon_path, recon_frames, fps=30, codec="libx264", quality=10)

    print("✅ Reconstruction completed. Videos should be playable in VLC, Chrome, etc.")
    break