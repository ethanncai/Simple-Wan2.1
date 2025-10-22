import torch
from torch.utils.data import Dataset, DataLoader
from ..utils.train_utils import load_and_preprocess_video

class OneShotVideoDataset(Dataset):
    def __init__(
        self,
        video_path,
        text,
        length=3200,
        frame_num=81,
        target_size=(480, 832)
    ):
        """
        Args:
            video_path (str): 视频文件路径
            text (str): 对应的文本描述（所有样本共享）
            length (int): 虚拟数据集长度
            frame_num (int): 视频帧数
            target_size (tuple): (H, W)
        """
        self.length = length
        self.text = text
        self.video_tensor = load_and_preprocess_video(
            video_path,
            frame_num=frame_num,
            target_size=target_size
        )  # (C, T, H, W)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # 忽略 index，返回相同的 (video, text)
        return self.video_tensor, self.text


if __name__ == "__main__":
    video_path = "/home/rapverse/workspace_junzhi/Wan2.1/t2v_832x480_cat_dancing_like_a_human_20251021_152920.mp4"
    prompt_text = "a cat dancing like a human"  # ←←← 你可以改成任意文本
    dataset = OneShotVideoDataset(
        video_path=video_path,
        text=prompt_text,
    )
    def collate_fn(batch):
        videos, texts = zip(*batch)
        return list(videos), list(texts)

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=lambda batch: ([v for v, t in batch], [t for v, t in batch])
    )

    # 测试 dataloader
    for i, (videos, texts) in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"  Video shape: {len(videos)}, {videos[0].shape}")   # (B, C, T, H, W)
        print(f"  Texts: {texts}")                # List[str]，长度为 batch_size
        if i >= 1:
            break