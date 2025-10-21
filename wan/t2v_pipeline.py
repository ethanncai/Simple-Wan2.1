# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
from contextlib import contextmanager

import torch
import torch.cuda.amp as amp
from tqdm import tqdm

from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanT2V:

    def __init__(
        self,
        model_hyperparam,
        checkpoint_dir,
        t5_cpu=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components (single-GPU only).

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU during encoding.
        """
        self.device = torch.device("cuda:0")
        self.model_hyperparam = model_hyperparam
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = model_hyperparam.num_train_timesteps
        self.param_dtype = model_hyperparam.param_dtype

        # Load T5 text encoder
        self.text_encoder = T5EncoderModel(
            text_len=model_hyperparam.text_len,
            dtype=model_hyperparam.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, model_hyperparam.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, model_hyperparam.t5_tokenizer),
            shard_fn=None  # No FSDP
        )

        # Load VAE
        self.vae_stride = model_hyperparam.vae_stride
        self.patch_size = model_hyperparam.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, model_hyperparam.vae_checkpoint),
            device=self.device
        )

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)
        self.model.to(self.device)

        self.sample_neg_prompt = model_hyperparam.sample_neg_prompt

    def generate(
        self,
        input_prompt,
        size=(1280, 720),
        frame_num=81,
        shift=5.0,
        sampling_steps=50,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
        offload_model=True
    ):
        r"""
        Generates video frames from text prompt using diffusion process (single-GPU).

        Args:
            input_prompt (`str`): Text prompt for content generation
            size (tuple[`int`], *optional*, defaults to (1280,720)): (width, height)
            frame_num (`int`, *optional*, defaults to 81): Number of frames (must be 4n+1)
            shift (`float`, *optional*, defaults to 5.0): Noise schedule shift
            sample_solver (`str`, *optional*, defaults to 'unipc'): Solver name
            sampling_steps (`int`, *optional*, defaults to 50): Diffusion steps
            guide_scale (`float`, *optional*, defaults to 5.0): CFG scale
            n_prompt (`str`, *optional*, defaults to ""): Negative prompt
            seed (`int`, *optional*, defaults to -1): Random seed
            offload_model (`bool`, *optional*, defaults to True): Offload to CPU to save VRAM

        Returns:
            torch.Tensor: Video tensor of shape (C, N, H, W)
        """
        F = frame_num
        target_shape = (
            self.vae.model.z_dim,
            (F - 1) // self.vae_stride[0] + 1,
            size[1] // self.vae_stride[1],
            size[0] // self.vae_stride[2]
        )

        seq_len = math.ceil(
            (target_shape[2] * target_shape[3]) /
            (self.patch_size[1] * self.patch_size[2]) *
            target_shape[1]
        )

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        # Encode prompts
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g
            )
        ]

        @contextmanager
        def noop():
            yield

        # Evaluation
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), noop():

            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False
            )
            sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps
            

            latents = noise
            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            for t in tqdm(timesteps):
                latent_model_input = latents
                timestep = torch.tensor([t], device=self.device)

                self.model.to(self.device)
                noise_pred_cond = self.model(latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g
                )[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            videos = self.vae.decode(x0)

        del noise, latents, sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()

        return videos[0]