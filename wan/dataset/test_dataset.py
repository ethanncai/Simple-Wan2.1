import torch
from torch.utils.data import Dataset, DataLoader
from ..utils.train_utils import load_and_preprocess_video, load_and_preprocess_video_segment


class OneShotVideoDataset(Dataset):
    def __init__(
        self,
        video_path,
        text,
        length=18,
        frame_num=81,
        target_size=(480, 832),
        seed=None
    ):
        """
        Args:
            video_path (str): 视频文件路径
            text (str): 对应的文本描述（所有样本共享）
            length (int): 虚拟数据集长度（即 epoch 中样本数）
            frame_num (int): 每次采样的连续帧数
            target_size (tuple): (H, W)
            seed (int, optional): 随机种子（用于可复现性）
        """
        self.video_path = video_path
        self.text = text
        self.length = length
        self.frame_num = frame_num
        self.target_size = target_size

        # Pre-check video length
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        self.total_frames = len(vr)
        if self.total_frames < self.frame_num:
            raise ValueError(f"Video has only {self.total_frames} frames, less than frame_num={frame_num}")

        self.max_start = self.total_frames - self.frame_num

        # Optional: set random seed for reproducibility in __getitem__
        if seed is not None:
            import random
            random.seed(seed)
            torch.manual_seed(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        import random
        start_idx = random.randint(0, self.max_start)

        # 加载视频段
        video_tensor = load_and_preprocess_video_segment(
            self.video_path,
            start_idx=start_idx,
            frame_num=self.frame_num,
            target_size=self.target_size
        )  # 输出 shape: (C, T, H, W)

        # 取第一帧
        first_frame = video_tensor[:, 0]  # shape: (C, H, W)

        # 返回 video, text, first_frame
        return video_tensor, random.choice(self.text), first_frame


if __name__ == "__main__":
    video_path = "/home/rapverse/workspace_junzhi/Wan2.1/t2v_832x480_cat_dancing_like_a_human_20251021_152920.mp4"
    prompt_text = ["prompt1","prompt2"]
    dataset = OneShotVideoDataset(
        video_path=video_path,
        text=prompt_text,
    )
    def collate_fn(batch):
        videos, texts, imgs = zip(*batch)
        return list(videos), list(texts), list(imgs)
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=collate_fn
    )

    # 测试 dataloader
    for i, (videos, texts, imgs) in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"  Video shape: {len(videos)}, {videos[0].shape}")   # (C, T, H, W)
        print(f"  Texts: {texts}")
        print(f"  First frame shape: {len(imgs)}, {imgs[0].shape}") # (C, H, W)
        if i >= 1:
            break
