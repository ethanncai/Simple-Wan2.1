import os
import csv
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader
import json
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import ipdb
from ..utils.train_utils import video_to_cthw, cthw_to_video,resize_but_retain_ratio
import torch
import torch.nn.functional as F
from pathlib import Path

class DatasetFromCSV(torch.utils.data.Dataset):
    """load video according to the csv file.

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(
        self,
        csv_path,
        target_w,
        target_h,
        target_frames=16,
        tries=10000,
        root=None,
    ):
        self.tries =tries
        self.target_frames = target_frames
        self.target_h = target_h
        self.target_w = target_w
        self.root = root
        self.T5_MAX_LEN = 512

        cache_file = Path(csv_path).with_suffix(".csv.cached")
        csv_mtime = os.path.getmtime(csv_path)

        if cache_file.exists():
            with open(cache_file, "rb") as f:
                cached_mtime, video_samples = pickle.load(f)
            if cached_mtime == csv_mtime:
                print(f"[Dataset] Loading cache {cache_file}")
                self.samples = video_samples
                return
            else:
                print(f"[Dataset] Cache not found")

        print("[Dataset] Reading CSV...")
        video_samples = []
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader, None)       
            for vid in reader:
                vid_name, vid_caption = vid[0], vid[1]
                vid_path = os.path.join(self.root, vid_name)
                if os.path.exists(vid_path):
                    video_samples.append([vid_path, vid_caption])
        print("[Dataset] Done Reading CSV.")

        # 写缓存
        with open(cache_file, "wb") as f:
            pickle.dump((csv_mtime, video_samples), f)
        print(f"[Dataset] Cache created {cache_file}")
        self.samples = video_samples

    def getitem(self, index):
        sample = self.samples[index]
        video_path = sample[0]
        text = sample[1]
        video_cthw = video_to_cthw(video_path)
        _,T,H,W = video_cthw.shape
        assert T >= self.target_frames
        
        # Handle Spacial
        video_cthw = resize_but_retain_ratio(video_cthw,(self.target_h,self.target_w))

        # Handle Augmentation

        # Handle Temporal
        start_idx = torch.randint(0, T - self.target_frames + 1, (1,)).item()
        video_cthw = video_cthw[:, start_idx:start_idx + self.target_frames, :, :]

        _,T_,H_,W_ = video_cthw.shape

        assert T_ == self.target_frames
        assert H_ == self.target_h
        assert W_ == self.target_w

        # Handle Textual
        if len(text) >= self.T5_MAX_LEN:
            text = text[:self.T5_MAX_LEN]

        return video_cthw, text

    def __getitem__(self, index):
        for _ in range(self.tries):
            try:
                return self.getitem(index)
            except Exception as e:
                print(e)
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':

    # data_path = '/mnt/data-oss/openvid_tiny/data/train/mini_openvid.csv'
    # root='/mnt/data-oss/openvid_tiny/video'
    data_path = '/mnt/data-oss/openvid/data/train/OpenVid-1M.csv'
    root='/mnt/data-oss/openvid/video'

    dataset = DatasetFromCSV(
        data_path,
        target_frames=81,
        target_h=280,
        target_w=832,
        root=root,
    )

    def collate_fn(batch):
        video_tensors = [item[0] for item in batch]   # list of video tensors
        texts = [item[1] for item in batch]           # list of strings
        return video_tensors, texts
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    for video_data in loader:
        video_list, text_list = video_data
        print(video_list[0].shape)
        print(text_list[0])
        cthw_to_video(video_list[0],"meow.mp4")
        break