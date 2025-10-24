# run_train.py
# DeepSpeed ZeRO-2 training for Wan2.1 T2V
# 修正版：log & save 都以 global step 为参照
# Author: 俊志 + GPT-5

import os
import gc
import math
import argparse
from datetime import datetime
import json

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import deepspeed
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

from wan.modules.model import WanModel
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae import WanVAE
from wan.dataset.test_dataset import OneShotVideoDataset
from wan.utils.train_utils import encode_video_and_text, load_weights


def get_timestamp():
    return datetime.now().strftime("%m%d%H%M")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Wan2.1 Text-to-Video Model (DeepSpeed ZeRO-2)")
    # core training args
    parser.add_argument("--checkpoint_dir", type=str, default="/home/rapverse/workspace_junzhi/Wan2.1/Wan2.1-T2V-1.3B")
    parser.add_argument("--output_dir", type=str, default="exp")
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=100000)
    parser.add_argument("--save_every", type=int, default=100)  # 默认以step计
    parser.add_argument("--log_every", type=int, default=5)    # 默认以step计
    parser.add_argument("--text_len", type=int, default=512)
    parser.add_argument("--frame_num", type=int, default=81)
    parser.add_argument("--size", type=str, default="832*480", help="W*H format")
    parser.add_argument("--vae_stride", nargs=3, type=int, default=[4, 8, 8])
    parser.add_argument("--patch_size", nargs=3, type=int, default=[1, 2, 2])
    parser.add_argument("--param_dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--t5_dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])

    # DeepSpeed config path (always using DS)
    parser.add_argument("--ds_config", type=str, default="ds_config_zero2.json", help="DeepSpeed config path")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None, help="Override GA steps in ds_config")

    # accept local_rank injected by DeepSpeed
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed by DeepSpeed/torchrun")

    return parser.parse_args()


def to_device_dtype(obj, device, dtype=None):
    if torch.is_tensor(obj):
        return obj.to(device=device, dtype=(dtype if dtype is not None else obj.dtype))
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_device_dtype(x, device, dtype) for x in obj)
    elif isinstance(obj, dict):
        return {k: to_device_dtype(v, device, dtype) for k, v in obj.items()}
    return obj


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    args = parse_args()

    # dtype map
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    param_dtype = dtype_map[args.param_dtype]
    t5_dtype = dtype_map[args.t5_dtype]

    # Parse resolution and patches
    W, H = map(int, args.size.split('*'))
    ps_t, ps_h, ps_w = args.patch_size

    # Demo dataset
    video_path = "/home/rapverse/workspace_junzhi/dataset/train/392-651388-007/head_color.mp4"
    prompt_text = [
        "POV of a pair of junzhi robot arm doing something",
        "POV of junzhi robot and it is doing something",
        "First person view of a junzhi brand robot arm and it is doing something",
        "POV of a pair of junzhi robot gripper",
        "First person view of a junzhi brand robot gripper and it is doing something",
    ]

    dataset = OneShotVideoDataset(video_path=video_path, text=prompt_text)
    def collate_fn(batch):
        videos, texts, imgs = zip(*batch)
        return list(videos), list(texts), list(imgs)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Load model
    print("Loading DiT...")
    model = WanModel()
    load_weights(model,'/home/rapverse/workspace_junzhi/Wan2.1/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors')
    model.train()

    # DeepSpeed initialize
    ds_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=filter(lambda p: p.requires_grad, model.parameters()),
        config=args.ds_config
    )

    device = ds_engine.device
    is_rank0 = (ds_engine.global_rank == 0)

    if is_rank0:
        print("\n" + "=" * 60)
        print("Training Configuration")
        print("=" * 60)
        config_dict = vars(args)
        for k, v in sorted(config_dict.items()):
            print(f"{k:<25}: {v}")
        print("-" * 60)
        total_params, trainable_params = count_parameters(model)
        print(f"{'Total Parameters':<25}: {total_params:,}")
        print(f"{'Trainable Parameters':<25}: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        print("=" * 60 + "\n")

    print("Loading VAE...")
    vae_checkpoint = os.path.join(args.checkpoint_dir, 'Wan2.1_VAE.pth')
    vae = WanVAE(vae_pth=vae_checkpoint, dtype=param_dtype, device=device)

    print("Loading T5...")
    t5_checkpoint = os.path.join(args.checkpoint_dir, 'models_t5_umt5-xxl-enc-bf16.pth')
    text_encoder = T5EncoderModel(
        text_len=args.text_len,
        dtype=t5_dtype,
        device=device,
        checkpoint_path=t5_checkpoint,
        tokenizer_path='google/umt5-xxl',
        shard_fn=None
    )
    print("Model loaded")

    # experiment dirs
    timestamp = get_timestamp()
    exp_name = f"exp_ds_{timestamp}"
    exp_dir = os.path.join(args.output_dir, exp_name)
    log_dir = os.path.join(args.log_dir, exp_name)
    if is_rank0:
        os.makedirs(exp_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir) if is_rank0 else None
    if is_rank0:
        print(f"Experiment directory: {exp_dir}")
        print(f"TensorBoard log dir: {log_dir}")

    global_step = 0
    total_steps_per_epoch = len(dataloader)

    # ----------------------- TRAIN LOOP -----------------------
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            videos, texts, imgs = batch
            B = len(videos)
            if B == 0:
                continue

            # --- encode video + text (on CPU, then move to GPU) ---
            with torch.no_grad():
                latents, context, img_latents = encode_video_and_text(
                    videos=videos,
                    imgs=imgs,
                    texts=texts,
                    vae=vae,
                    text_encoder=text_encoder,
                    device=device,
                    param_dtype=torch.bfloat16
                )

            x_1 = torch.stack(latents, dim=0).to(device, dtype=param_dtype)
            context = to_device_dtype(context, device, dtype=param_dtype)

            _, T_z, H_z, W_z = x_1.shape[1:]
            assert T_z % ps_t == 0 and H_z % ps_h == 0 and W_z % ps_w == 0, \
                f"Latent dims not divisible by patch_size: got T={T_z},H={H_z},W={W_z} vs ps={args.patch_size}"
            seq_len = (T_z // ps_t) * (H_z // ps_h) * (W_z // ps_w)

            # fractional timestep [0,1]
            t_frac = torch.rand(B, device=device, dtype=param_dtype)
            t_frac_exp = t_frac.view(B, 1, 1, 1, 1)

            # x_0 = random noise
            x_0 = torch.randn_like(x_1, device=device, dtype=param_dtype)

            # mix latents and noise
            x_t = (1.0 - t_frac_exp) * x_1 + t_frac_exp * x_0

            # target = x_0 - x_1
            target = x_0 - x_1

            # model expects timestep [0,1000]
            t_model = (t_frac * 1000.0).to(device=device, dtype=param_dtype)

            # --- forward + loss ---
            ds_engine.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                velocity_pred_list = ds_engine(x_t, t=t_model, context=context, seq_len=seq_len, img=img_latents)
                if isinstance(velocity_pred_list, (list, tuple)):
                    velocity_pred = torch.stack(velocity_pred_list, dim=0)
                else:
                    velocity_pred = velocity_pred_list

                loss = F.mse_loss(velocity_pred, target)

            ds_engine.backward(loss)
            ds_engine.step()

            loss_val = float(loss.detach().item())
            epoch_loss += loss_val
            num_batches += 1
            global_step += 1

            # --- logging by global step ---
            if is_rank0 and (global_step % args.log_every == 0):
                writer.add_scalar("Loss/train", loss_val, global_step)
                print(f"[Step {global_step}] Loss: {loss_val:.6f}")

            # --- checkpoint save by global step ---
            if is_rank0 and (global_step % args.save_every == 0):
                ckpt_dir = os.path.join(exp_dir, f"ds_step_{global_step}")
                ds_engine.save_checkpoint(ckpt_dir, tag=f"global_step_{global_step}")

            # cleanup per batch
            del x_t, x_0, x_1, target, loss
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # --- optional epoch summary (still useful for logging) ---
        avg_loss = epoch_loss / max(1, num_batches)
        if is_rank0:
            writer.add_scalar("Loss/epoch_avg", avg_loss, global_step)
            print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.6f}")

    # close writer
    if is_rank0 and writer is not None:
        writer.close()
    if is_rank0:
        print("Training completed.")


if __name__ == "__main__":
    main()
