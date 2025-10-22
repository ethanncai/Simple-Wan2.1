# Author: Junzhi.Cai
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

from wan.modules.model import WanModel
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae import WanVAE
import math
from typing import List, Tuple
import torch

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
import torch

def print_gpu_memory(prefix):
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    current_memory = torch.cuda.memory_allocated()
    max_memory = torch.cuda.max_memory_allocated()
    total_memory = torch.cuda.get_device_properties(0).total_memory

    def format_bytes(bytes_val):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_val < 1024.0:
                return f"{bytes_val:.2f} {unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.2f} TB"

    print(f"GPU Memory [{prefix}] - Current: {format_bytes(current_memory)}, "
          f"Peak: {format_bytes(max_memory)}, "
          f"Total: {format_bytes(total_memory)}")

def encode_video_and_text(
    videos: List[torch.Tensor],
    texts: List[str],
    vae,
    text_encoder,
    vae_stride: Tuple[int, int, int] = (4, 8, 8),
    patch_size: Tuple[int, int, int] = (1, 2, 2),
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    param_dtype: torch.dtype = torch.bfloat16,
    debug=False
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    assert len(videos) == len(texts), "Batch size mismatch between videos and texts"

    # Move VAE to device (assuming it's not already there)

    # ======================
    # Optional: log latent seq_len (for debugging)
    # ======================
    C, T, H, W = videos[0].shape
    target_shape = (
        vae.model.z_dim,
        (T - 1) // vae_stride[0] + 1,  # T_z
        H // vae_stride[1],            # H_z
        W // vae_stride[2]             # W_z
    )
    seq_len = math.ceil(
        (target_shape[2] * target_shape[3]) /
        (patch_size[1] * patch_size[2]) *
        target_shape[1]
    )
    # print(f"[Debug] Latent shape: {target_shape}, seq_len: {seq_len}")

    # ======================
    # VAE Encoding
    # ======================
    videos_on_device = [v.to(device) for v in videos]
    if debug: print_gpu_memory('before moving vae to gpu')
    vae.model.to(device)
    if debug: print_gpu_memory('moved vae to gpu')
    with torch.no_grad(),torch.autocast(device_type="cuda", dtype=param_dtype):
        latents = vae.encode(videos_on_device)  # List[(C_z, T_z, H_z, W_z)]
    vae.model.to("cpu")
    if debug: print_gpu_memory('moved vae to cpu')
    torch.cuda.empty_cache()

    # ======================
    # Text Encoding
    # ======================
    if debug: print_gpu_memory('before moving t5 to gpu')
    text_encoder.model.to(device)
    if debug: print_gpu_memory('moved t5 to gpu')
    with torch.no_grad(),torch.autocast(device_type="cuda", dtype=param_dtype):
        context = text_encoder(texts, device)  # List[(L, D)]
    text_encoder.model.to("cpu")
    if debug: print_gpu_memory('moved t5 to cpu')
    torch.cuda.empty_cache()  # optional: free GPU memory if text encoder is large

    # Ensure context tensors are on device and correct dtype
    context = [t.to(device).to(param_dtype) for t in context]
    return latents, context