# run_train.py
# DeepSpeed ZeRO-2 training for Wan2.1 T2V
# - Fixes OOM pitfalls (multi-output stacking, seq_len calc, autocast)
# - Uses VAE/T5 on CPU for encoding; moves latents/context to GPU
# - Always runs with DeepSpeed; accepts --local_rank injected by launcher

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
from wan.utils.train_utils import encode_video_and_text


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
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=1)
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


def export_ds_to_hf(ds_ckpt_dir, output_hf_dir, original_config_path):
    """Export DeepSpeed ZeRO-2 checkpoint to HuggingFace-style model."""
    if not os.path.exists(ds_ckpt_dir):
        print(f"Skip export: {ds_ckpt_dir} not found.")
        return

    print(f"Exporting DeepSpeed checkpoint to HF format: {output_hf_dir}")
    try:
        # Load empty model structure
        model = WanModel.from_pretrained(original_config_path)
        # Reconstruct full state dict from ZeRO shards
        state_dict = load_state_dict_from_zero_checkpoint(model, ds_ckpt_dir)
        # Save
        os.makedirs(output_hf_dir, exist_ok=True)
        torch.save(state_dict, os.path.join(output_hf_dir, "pytorch_model.bin"))
        # Copy config if exists
        config_src = os.path.join(original_config_path, "config.json")
        if os.path.exists(config_src):
            import shutil
            shutil.copy(config_src, os.path.join(output_hf_dir, "config.json"))
        print(f"HF model saved to {output_hf_dir}")
    except Exception as e:
        print(f"Export failed: {e}")


def main():
    args = parse_args()

    # dtype map
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    param_dtype = dtype_map[args.param_dtype]
    t5_dtype = dtype_map[args.t5_dtype]

    # Parse resolution and patches
    W, H = map(int, args.size.split('*'))
    ps_t, ps_h, ps_w = args.patch_size

    # Demo dataset: one sample
    video_path = "/home/rapverse/workspace_junzhi/dataset/mini_train/367-648981-002/head_color.mp4"
    prompt_text = "a pair of anker robot arm picking up bread and place it down"

    dataset = OneShotVideoDataset(video_path=video_path, text=prompt_text)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Load model, VAE, T5
    print("Loading DiT...")
    model = WanModel.from_pretrained(args.checkpoint_dir)
    model.train()

    # DeepSpeed initialize (always enabled)
    ds_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=filter(lambda p: p.requires_grad, model.parameters()),
        config=args.ds_config
    )

    device = ds_engine.device
    is_rank0 = (ds_engine.global_rank == 0)

    # Print training config & model stats (only rank0)
    if is_rank0:
        print("\n" + "="*60)
        print("raining Configuration")
        print("="*60)
        config_dict = vars(args)
        for k, v in sorted(config_dict.items()):
            print(f"{k:<25}: {v}")
        print("-"*60)
        total_params, trainable_params = count_parameters(model)
        print(f"{'Total Parameters':<25}: {total_params:,}")
        print(f"{'Trainable Parameters':<25}: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        print("="*60 + "\n")

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

    # BF16 autocast for activations
    use_bf16_autocast = (args.param_dtype == "bfloat16")
    autocast_cm = torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16_autocast)

    # Experiment dirs
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

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # Unpack (video, text) pairs
            if isinstance(batch, list) and len(batch) > 0 and isinstance(batch[0], tuple):
                videos = [v for v, t in batch]
                texts = [t for v, t in batch]
            else:
                videos, texts = batch

            B = len(videos)
            if B == 0:
                continue

            # Encode on CPU
            with torch.no_grad():
                latents, context = encode_video_and_text(
                    videos=videos,
                    texts=texts,
                    vae=vae,
                    text_encoder=text_encoder,
                    vae_stride=tuple(args.vae_stride),
                    patch_size=tuple(args.patch_size),
                    device=torch.device("cuda"),
                    param_dtype=torch.bfloat16
                )

            x0_batch = torch.stack(latents, dim=0).to(device, dtype=param_dtype)
            context = to_device_dtype(context, device, dtype=param_dtype)

            _, T_z, H_z, W_z = x0_batch.shape[1:]
            assert T_z % ps_t == 0 and H_z % ps_h == 0 and W_z % ps_w == 0, \
                f"Latent dims not divisible by patch_size: got T={T_z},H={H_z},W={W_z} vs ps={args.patch_size}"
            seq_len = (T_z // ps_t) * (H_z // ps_h) * (W_z // ps_w)

            t = torch.rand(B, device=device, dtype=param_dtype)
            noise = torch.randn_like(x0_batch)
            x_t = (1 - t.view(B, 1, 1, 1, 1)) * x0_batch + t.view(B, 1, 1, 1, 1) * noise
            target_velocity = noise - x0_batch

            ds_engine.optimizer.zero_grad(set_to_none=True)
            with autocast_cm:
                velocity_pred_list = ds_engine(x_t, t=t, context=context, seq_len=seq_len)
                velocity_pred = torch.stack(velocity_pred_list, dim=0)
                loss = F.mse_loss(velocity_pred, target_velocity)

            ds_engine.backward(loss)
            ds_engine.step()

            loss_val = float(loss.detach().item())
            epoch_loss += loss_val
            num_batches += 1
            global_step += 1

            if is_rank0 and (global_step % args.log_every == 0):
                writer.add_scalar("Loss/train", loss_val, global_step)
                # Print epoch, global step, and step-in-epoch
                print(f"[Epoch {epoch+1}/{args.num_epochs}][Step {batch_idx+1}/{total_steps_per_epoch}] "
                      f"GlobalStep {global_step}, Loss: {loss_val:.6f}")

            # Demo: one batch per epoch
            # break

        avg_loss = epoch_loss / max(1, num_batches)
        if is_rank0:
            writer.add_scalar("Loss/epoch", avg_loss, epoch)
            print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.6f}")

        # Save DeepSpeed checkpoint (all ranks)
        if (epoch + 1) % args.save_every == 0:
            ckpt_dir = os.path.join(exp_dir, f"ds_epoch_{epoch+1}")
            ds_engine.save_checkpoint(ckpt_dir, tag=f"global_step_{global_step}")
            # if is_rank0:
            #     print(f"DeepSpeed sharded checkpoint saved to {ckpt_dir}")
            #     # Export to HF format immediately
            #     hf_dir = os.path.join(exp_dir, f"hf_epoch_{epoch+1}")
            #     export_ds_to_hf(ckpt_dir, hf_dir, args.checkpoint_dir)

        # Cleanup
        del x0_batch, x_t, noise, target_velocity, loss
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final export (optional, if not saved on last epoch)
    # if is_rank0 and (args.num_epochs % args.save_every != 0):
    #     last_ckpt = os.path.join(exp_dir, f"ds_epoch_{args.num_epochs}")
    #     if os.path.exists(last_ckpt):
    #         hf_dir = os.path.join(exp_dir, f"hf_epoch_{args.num_epochs}")
    #         export_ds_to_hf(last_ckpt, hf_dir, args.checkpoint_dir)

    if is_rank0 and writer is not None:
        writer.close()
    if is_rank0:
        print("Training completed.")


if __name__ == "__main__":
    main()