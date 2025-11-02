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
import numpy as np
import cv2
from decord import VideoReader, cpu

import subprocess
import json

def get_video_info(video_path: str):
    """
    使用 ffprobe 获取视频的元信息，不解码视频帧
    """
    cmd = [
        "ffprobe",
        "-v", "error",                     # 只输出错误信息
        "-select_streams", "v:0",          # 只选择视频流
        "-show_entries", "stream=width,height,nb_frames,duration,avg_frame_rate",
        "-of", "json",                     # 输出为 JSON 格式
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    info = json.loads(result.stdout)
    
    assert "streams" in info, f"未检测到视频流 in {video_path}"
    if not info["streams"]:
        raise ValueError(f"未检测到视频流 in {video_path}")

    stream = info["streams"][0]
    fps = eval(stream.get("avg_frame_rate", "0")) if stream.get("avg_frame_rate") != "0/0" else 0
    frames = int(stream["nb_frames"]) if stream.get("nb_frames", "0").isdigit() else None

    return {
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "duration": float(stream.get("duration", 0.0)),
        "fps": fps,
        "frames": frames
    }

def video_to_cthw(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)

    if total_frames == 0:
        raise ValueError(f"Video {video_path} has no frames.")
    frames = vr.get_batch(range(total_frames)).asnumpy()
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()  # (T, H, W, C) -> (T, C, H, W)
    frames = frames / 127.5 - 1.0
    video_tensor = frames.permute(1, 0, 2, 3)  # (C, T, H, W)
    return video_tensor.contiguous()


import torch
import numpy as np
import imageio.v3 as iio   # imageio >=2.9
from typing import Union, Tuple

def cthw_to_video(video_tensor: torch.Tensor,
                  output_path: str,
                  fps: int = 16,
                  crf: int = 18,          # 视觉无损，越小越清晰
                  macro_block_size: int = 16) -> None:

    if video_tensor.ndim != 4 or video_tensor.shape[0] != 3:
        raise ValueError("Input must be (3, T, H, W)")

    frames = ((video_tensor.permute(1, 2, 3, 0) + 1.0) * 127.5).clamp(0, 255)
    frames: np.ndarray = frames.cpu().numpy().astype(np.uint8)  # (T, H, W, 3)

    T, H, W, _ = frames.shape

    pad_h = (macro_block_size - H % macro_block_size) % macro_block_size
    pad_w = (macro_block_size - W % macro_block_size) % macro_block_size
    if pad_h or pad_w:
        frames = np.pad(frames,
                        ((0, 0), (0, pad_h), (0, pad_w), (0, 0)),
                        mode='edge')

    iio.imwrite(output_path,
                frames,
                fps=fps,
                codec='libx264',
                pixelformat='yuv420p',
                output_params=['-crf', str(crf),
                               '-preset', 'medium'])

def stretchly_resize(clip, target_size):
    C, T, H_in, W_in = clip.shape
    H_out, W_out = target_size
    
    # 将 (C, T, H, W) 重排为 (T, C, H, W) 以便 interpolate 处理
    clip = clip.permute(1, 0, 2, 3)  # (T, C, H_in, W_in)
    
    # 使用 interpolate 进行 resize，mode='bilinear' 适用于图像
    resized_clip = F.interpolate(clip, size=(H_out, W_out), mode='bilinear', align_corners=False)
    
    # 恢复原始维度顺序 (C, T, H_out, W_out)
    resized_clip = resized_clip.permute(1, 0, 2, 3)
    
    return resized_clip

def resize_but_retain_ratio(clip, target_size):
    """
    Performs a "cover" resize: scales the video while preserving aspect ratio so that
    the entire target area is covered, then center-crops to match the target aspect ratio,
    and finally resizes to exact target resolution.

    Args:
        clip (torch.Tensor): Video of shape (C, T, H, W), dtype float (any range).
        target_size (tuple): (H_out, W_out), e.g., (480, 832)

    Returns:
        torch.Tensor: Video of shape (C, T, H_out, W_out)
    """
    C, T, H_in, W_in = clip.shape
    H_out, W_out = target_size

    # Compute target aspect ratio
    target_aspect = W_out / H_out
    input_aspect = W_in / H_in

    # Step 1: Determine scale factor to ensure full coverage ("cover" mode)
    if input_aspect > target_aspect:
        # Input is wider → scale based on height
        scale = H_out / H_in
    else:
        # Input is taller or same ratio → scale based on width
        scale = W_out / W_in

    # Apply scale to get intermediate size
    new_H = int(round(H_in * scale))
    new_W = int(round(W_in * scale))

    # Step 2: Temporarily reshape to (C*T, 1, H, W) for interpolation
    # But F.interpolate expects (N, C, H, W), so we merge C and T into batch dim
    clip_reshaped = clip.view(C * T, 1, H_in, W_in)  # (C*T, 1, H, W)
    clip_scaled = F.interpolate(
        clip_reshaped,
        size=(new_H, new_W),
        mode='bilinear',
        align_corners=False,
        antialias=True
    )  # (C*T, 1, new_H, new_W)
    clip_scaled = clip_scaled.view(C, T, new_H, new_W)

    # Step 3: Center crop to match target aspect ratio in the scaled space
    if new_H >= H_out and new_W >= W_out:
        if abs(new_W / new_H - target_aspect) < 1e-6:
            clip_cropped = clip_scaled
        else:
            if new_W / new_H > target_aspect:
                # Too wide: fix height, adjust width
                crop_h = new_H
                crop_w = int(round(crop_h * target_aspect))
            else:
                # Too tall: fix width, adjust height
                crop_w = new_W
                crop_h = int(round(crop_w / target_aspect))

            crop_h = min(crop_h, new_H)
            crop_w = min(crop_w, new_W)

            top = (new_H - crop_h) // 2
            left = (new_W - crop_w) // 2
            clip_cropped = clip_scaled[:, :, top:top + crop_h, left:left + crop_w]
    else:
        # Scaled video smaller than target in one dimension → skip crop
        clip_cropped = clip_scaled

    # Step 4: Final resize to exact target size
    C2, T2, H_c, W_c = clip_cropped.shape
    clip_reshaped_final = clip_cropped.view(C2 * T2, 1, H_c, W_c)
    clip_final = F.interpolate(
        clip_reshaped_final,
        size=(H_out, W_out),
        mode='bilinear',
        align_corners=False,
        antialias=True
    ).view(C2, T2, H_out, W_out)

    return clip_final



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
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from PIL import Image

def load_and_preprocess_image(image_path, target_size=(480, 832)):
    """
    Args:
        image_path (str): 图片文件路径
        target_size (tuple): (H, W)

    Returns:
        torch.Tensor: shape (C, 1, H, W), dtype float32, range [-1, 1]
    """
    H_target, W_target = target_size

    transform = Compose([
        ToTensor(),  # (H, W, C) -> (C, H, W)，范围 [0, 1]
        Resize((H_target, W_target), antialias=True),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # -> [-1, 1]
    ])

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).float()  # (3, H, W)
    tensor = tensor.unsqueeze(1)     # (3, 1, H, W)

    return tensor


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
    clip_encoder,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    param_dtype: torch.dtype = torch.bfloat16,
    debug=False
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    assert len(videos) == len(texts), "Batch size mismatch between videos and texts"


    # Move VAE to device (assuming it's not already there)
    # ======================
    # VAE Encoding
    # ======================
    imgs = [v[:,:1,...] for v in videos] # img -> [C1HW,]

    videos_on_device = [v.to(device) for v in videos]
    imgs_on_device = [v.to(device) for v in imgs]
    imgs_on_device = [v.repeat(1,4,1,1) for v in imgs_on_device]
    assert videos_on_device[0].dtype == imgs_on_device[0].dtype
    if debug: print_gpu_memory('before moving vae to gpu')
    vae.model.to(device)
    if debug: print_gpu_memory('moved vae to gpu')
    with torch.no_grad(),torch.autocast(device_type="cuda", dtype=param_dtype):
        latents = vae.encode(videos_on_device)  # List[(C_z, T_z, H_z, W_z)]
        img_latents = vae.encode(imgs_on_device)
        assert img_latents[0].shape[1] == 1,f"{img_latents[0].shape}"
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

    # ======================
    # Clip Encoding
    # ======================
    with torch.no_grad(),torch.autocast(device_type="cuda", dtype=param_dtype):
        clip_feat = [clip_encoder(v) for v in imgs_on_device]
     # B,768

    # Ensure context tensors are on device and correct dtype
    context = [t.to(param_dtype) for t in context]
    latents = [t.to(param_dtype) for t in latents]
    img_latents = [t.to(param_dtype) for t in img_latents]
    return latents, context, img_latents, clip_feat

def create_alternating_forward(*block_lists, grad_checkpoint):
    max_length = max(len(block_list) for block_list in block_lists) if block_lists else 0
    call_sequence = []
    for i in range(max_length):
        for block_list in block_lists:
            if i < len(block_list):
                call_sequence.append(block_list[i])
    
    def alter_pass(x, c):
        for transformer_block in call_sequence:
            # 关键改动：禁用 reentrant
            x = grad_checkpoint(transformer_block, x, c, use_reentrant=False)
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


import torch
from typing import List

def random_drop(x: List[torch.Tensor], drop_prob: float = 0.1):
    if drop_prob <= 0 or len(x) == 0:
        return x

    B = len(x)
    device = x[0].device
    dtype = x[0].dtype

    keep_mask = (torch.rand(B, device=device) > drop_prob).to(dtype)
    out = []
    for i in range(B):
        if keep_mask[i] == 0:
            out.append(torch.zeros_like(x[i], dtype=dtype, device=device))
        else:
            out.append(x[i])
    return out

def compute_dispersion_moduli(block_lists: List[List]) -> List[int]:

    if not block_lists:
        return []
    
    lengths = [len(blk) for blk in block_lists]
    max_len = max(lengths)
    
    moduli = []
    for L in lengths:
        if L == 0:
            moduli.append(1)  # avoid div by zero; won't be used anyway
        else:
            # 目标：大约每 (max_len / L) 步触发一次
            m = max(1, round(max_len / L))
            moduli.append(m)
    return moduli