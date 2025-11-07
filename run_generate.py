# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import random

import torch
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.utils import cache_image, cache_video, str2bool


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an image or video from a text prompt using Wan (single-GPU mode)"
    )
    parser.add_argument(
        "--gen_task",
        type=str,
        default="ia2v",
        choices=["i2v", "ia2v"],
        help="The generation task to run."
    )
    parser.add_argument(
        "--img_path",
        type=str,
        required=True,
        help="initial frame")
    parser.add_argument(
        "--action_metadata_path",
        type=str,
        default=None,
        help="this point to a directory, which contains ex.json in.json h5 file")
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video.")
    parser.add_argument(
        "--frame_num",
        type=int,
        default=81,
        help="How many frames to sample. Should be 4n+1 (or 1 for t2i)."
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="Path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload model to CPU after forward to save VRAM."
    )
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to keep T5 on CPU during encoding."
    )
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="File to save the generated image or video to."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate from."
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="Random seed for generation."
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=50,
        help="Number of sampling steps."
    )
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers."
    )
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale."
    )

    args = parser.parse_args()
    # _validate_args(args)
    return args


def _init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def generate(args):
    _init_logging()

    model_hyperparam = WAN_CONFIGS[args.task]
    logging.info(f"Generation job args: {args}")
    logging.info(f"Model config: {model_hyperparam}")

    assert "t2v" in args.task, "This script currently only supports 't2v' task in simplified version."
    logging.info(f"Input prompt: {args.prompt}")

    logging.info("Creating WanT2V pipeline.")
    wan_t2v_pipeline = wan.GenPipeline(
        gen_task=args.gen_task,
        model_hyperparam=model_hyperparam,
        checkpoint_dir=args.ckpt_dir,
        t5_cpu=args.t5_cpu,
    )

    logging.info("Generating video...")
    video = wan_t2v_pipeline.generate(
        img_path=args.img_path,
        action_metadata_path=args.action_metadata_path,
        input_prompt=args.prompt,
        size=SIZE_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.base_seed,
        offload_model=args.offload_model
    )

    if args.save_file is None:
        formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        formatted_prompt = args.prompt.replace(" ", "_").replace("/", "_")[:50]
        suffix = '.mp4'  # Only t2v supported
        args.save_file = f"t2v_{args.size.replace('*', 'x')}_{formatted_prompt}_{formatted_time}{suffix}"

    logging.info(f"Saving generated video to {args.save_file}")
    cache_video(
        tensor=video[None],
        save_file=args.save_file,
        fps=5,
        nrow=1,
    )
    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)