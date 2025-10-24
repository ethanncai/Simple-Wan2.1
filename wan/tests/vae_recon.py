import os
import torch
from decord import VideoReader, cpu
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import imageio.v3 as iio
import numpy as np

from ..modules.vae import WanVAE

def load_video_frames(video_path, frame_num=81, target_size=(480, 832)):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    if total_frames == 0:
        raise ValueError(f"Video {video_path} has 0 frames.")
    
    indices = torch.linspace(0, total_frames - 1, frame_num).long().tolist()
    frames = vr.get_batch(indices).asnumpy()  # (F, H_orig, W_orig, C)

    H_target, W_target = target_size
    # 确保宽高为偶数（H.264 要求）
    if H_target % 2 != 0:
        H_target += 1
    if W_target % 2 != 0:
        W_target += 1

    transform = Compose([
        ToTensor(),  # (H, W, C) -> (C, H, W), [0,1]
        Resize((H_target, W_target), antialias=True),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # -> [-1, 1]
    ])

    frame_tensors = []
    for i in range(frames.shape[0]):
        img = Image.fromarray(frames[i])
        frame_tensors.append(transform(img))
    video_tensor = torch.stack(frame_tensors, dim=1)  # (C, F, H, W)
    return video_tensor, (H_target, W_target)

def tensor_to_uint8_rgb(video_tensor):
    """
    将 [-1,1] 的 (C, F, H, W) tensor 转为 (F, H, W, C) uint8 RGB numpy array
    """
    video_tensor = video_tensor.clamp(-1, 1)
    video_tensor = (video_tensor + 1) / 2  # [-1,1] -> [0,1]
    video_tensor = video_tensor.permute(1, 2, 3, 0)  # (F, H, W, C)
    return (video_tensor * 255).byte().cpu().numpy()

# ======================
# 配置
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_dir = "/home/rapverse/workspace_junzhi/Wan2.1/Wan2.1-T2V-1.3B"
vae_checkpoint = os.path.join(checkpoint_dir, 'Wan2.1_VAE.pth')

frame_num = 81
size_str = "832*480"
W, H = map(int, size_str.split('*'))

# 替换为你的视频路径
video_path = "/home/rapverse/workspace_junzhi/Wan2.1/test_video.mp4"
output_recon_path = "./reconstructed_video.mp4"
output_orig_path = "./original_resized.mp4"

# ======================
# 加载视频（自动对齐到偶数尺寸）
# ======================
print("Loading and preprocessing video...")
video_tensor, (H_adj, W_adj) = load_video_frames(
    video_path,
    frame_num=frame_num,
    target_size=(H, W)
)
print(f"Adjusted target size: {W_adj}x{H_adj}")
print(f"Input video tensor shape: {video_tensor.shape}")

# ======================
# 加载 VAE
# ======================
print("Loading WanVAE...")
vae = WanVAE(
    vae_pth=vae_checkpoint,
    dtype=torch.bfloat16,
    device=device
)

# ======================
# Encode + Decode
# ======================
video_tensor = video_tensor.to(device)
with torch.no_grad():
    print("Encoding...")
    print(video_tensor.dtype)
    latents = vae.encode([video_tensor])
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
iio.imwrite(output_orig_path, orig_frames, fps=8, codec="libx264", quality=10)

print(f"Saving reconstructed video to {output_recon_path}...")
iio.imwrite(output_recon_path, recon_frames, fps=8, codec="libx264", quality=10)

print("✅ Reconstruction completed. Videos should be playable in VLC, Chrome, etc.")