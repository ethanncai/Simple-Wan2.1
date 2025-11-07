import argparse
import random
import os
from decord import VideoReader, cpu
from PIL import Image
import numpy as np

def extract_random_frame(video_path, output_path,first_frame):
    # 检查文件是否存在
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # 读取视频
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)

    if total_frames == 0:
        raise ValueError(f"Video {video_path} has 0 frames.")

    # 随机选取帧索引
    if first_frame:
        idx = 0
    else:
        idx = random.randint(0, total_frames - 1)

    # 获取该帧图像
    frame = vr[idx].asnumpy()  # shape: (H, W, C), dtype=uint8
    img = Image.fromarray(frame)

    # 保存图片
    img.save(output_path)
    print(f"Saved frame {idx} / {total_frames} from {video_path} -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract a random frame from a video.")
    parser.add_argument("-v", "--video", required=True, help="Path to input video file.")
    parser.add_argument("-i", "--image", required=True, help="Path to output image file.")
    parser.add_argument("--first_frame", type=bool, default=True, help="Path to output image file.")
    args = parser.parse_args()

    extract_random_frame(args.video, args.image, args.first_frame)
