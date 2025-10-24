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

from decord import VideoReader, cpu
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from PIL import Image
import torch

def load_and_preprocess_video_segment(
    video_path,
    start_idx,
    frame_num=81,
    target_size=(480, 832)
):
    """
    Load a continuous segment of `frame_num` frames starting from `start_idx`.

    Returns:
        torch.Tensor: shape (C, T, H, W), dtype float32, range [-1, 1]
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)

    if total_frames < frame_num:
        raise ValueError(f"Video {video_path} has only {total_frames} frames, less than required {frame_num}.")

    # Ensure start_idx is valid
    max_start = total_frames - frame_num
    if start_idx < 0 or start_idx > max_start:
        raise ValueError(f"Invalid start_idx {start_idx} for video with {total_frames} frames and segment length {frame_num}.")

    indices = list(range(start_idx, start_idx + frame_num))  # [s, s+1, ..., s+frame_num-1]
    frames = vr.get_batch(indices).asnumpy()  # (frame_num, H_orig, W_orig, C)

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

def create_alternating_forward(*block_lists, grad_checkpoint):
    max_length = max(len(block_list) for block_list in block_lists) if block_lists else 0
    call_sequence = []
    for i in range(max_length):
        for block_list in block_lists:
            if i < len(block_list):
                call_sequence.append(block_list[i])
    
    def alter_pass(x, c):
        for transformer_block in call_sequence:
            x = grad_checkpoint(transformer_block, x, c)
        return x
    
    return alter_pass

import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch
import torch.nn as nn
from safetensors.torch import load_file

def load_weights(model: nn.Module, checkpoint_path: str, device="cpu"):
    """
    Native PyTorch 权重加载，支持 .pt/.bin 和 .safetensors，忽略缺失和形状不匹配的权重，并打印加载情况。
    """
    # 1. 如果模型是 meta tensor，先分配实际显存
    try:
        model = model.to(device)
    except RuntimeError as e:
        if "meta tensor" in str(e):
            print("Meta tensor detected, using to_empty()...")
            model = model.to_empty(device=device)

    # 2. 加载 checkpoint
    if checkpoint_path.endswith(".safetensors"):
        checkpoint = load_file(checkpoint_path, device=device)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    # 3. 获取模型 state_dict
    model_dict = model.state_dict()

    # 统计信息
    total_params = sum(p.numel() for p in model.parameters())
    loaded_params = 0
    missing_keys = []
    unexpected_keys = []

    # 4. 过滤 checkpoint，只加载 key 匹配且 shape 匹配的权重
    filtered_dict = {}
    for k, v in checkpoint.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                filtered_dict[k] = v
                loaded_params += v.numel()
            else:
                print(f"[Mismatch] {k}: checkpoint {v.shape} vs model {model_dict[k].shape}, skipping")
                missing_keys.append(k)
        else:
            print(f"[Unexpected] {k} not in model, skipping")
            unexpected_keys.append(k)

    # 5. 找出模型中缺失的 key
    for k in model_dict.keys():
        if k not in filtered_dict:
            missing_keys.append(k)

    # 6. 更新权重
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)

    print(f"=== Load Summary ===")
    print(f"Total model parameters: {total_params:,}")
    print(f"Loaded parameters: {loaded_params:,} ({loaded_params/total_params*100:.2f}%)")
    print(f"Missing keys (randomly initialized): {len(missing_keys)}")
    print(f"Unexpected keys in checkpoint (ignored): {len(unexpected_keys)}")

    return model, missing_keys, unexpected_keys
