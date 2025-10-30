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
    
    if not info["streams"]:
        raise ValueError("未检测到视频流")

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

# 示例
info = get_video_info("/home/rapverse/workspace_junzhi/Simple-Wan2.1/meow.mp4")
print(info)
