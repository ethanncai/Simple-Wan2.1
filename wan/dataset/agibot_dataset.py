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
from ..utils.train_utils import video_to_cthw, cthw_to_video,resize_but_retain_ratio,get_video_info
import torch
import torch.nn.functional as F
from pathlib import Path
import pickle

class AgibotDataset(torch.utils.data.Dataset):
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
        target_frames=81,
        sample_interval=6, # means slow down 6x
        tries=10000,
        agibot_dataset_path=None,
        agibot_task_info_path=None,
        
    ):
        # 这个类只在加载的时候进行完整性检查。indexing的时候不对样本是否corrupt进行检查
        self.tries =tries
        self.target_frames = target_frames
        self.target_h = target_h
        self.target_w = target_w
        self.agibot_dataset_path = agibot_dataset_path
        self.agibot_task_info_path = agibot_task_info_path
        self.T5_MAX_LEN = 512
        self.raw_frame_required = self.target_frames * sample_interval

        print("[Dataset] Indexing Dataset...")
        self.sample_paths = list(os.listdir(self.agibot_dataset_path))

    def getitem(self, index):
        sample_dir_name = self.samples[index]
        sample_dir_path = os.path.join(self.agibot_dataset_path, sample_dir_name)
        
        video_path = os.path.join(sample_dir_path,
                                    "head_color.mp4")
        ex_json_path = os.path.join(sample_dir_path,
                                    "head_extrinsic_params_aligned.json")
        in_json_path = os.path.join(sample_dir_path,
                                    "head_intrinsic_params.json")
        h5_json_path = os.path.join(sample_dir_path,
                                    "proprio_stats.h5")


        video_path = sample_dir_name[0]
        text = sample_dir_name[1]
        video_cthw = video_to_cthw(video_path)
        _,T,H,W = video_cthw.shape
        
        if T < self.target_frames:
            pad = self.target_frames - T
            video_cthw = torch.cat(
                [video_cthw, video_cthw[:, -1:].repeat_interleave(pad, dim=1)],
                dim=1
            )
            T = self.target_frames

        # Handle Spacial
        video_cthw = resize_but_retain_ratio(video_cthw,(self.target_h,self.target_w))

        # Handle Augmentation

        # Handle Temporal
        start_idx = torch.randint(0, T - self.target_frames + 1, (1,)).item()
        video_cthw = video_cthw[:, start_idx:start_idx + self.target_frames, :, :]

        _,T_,H_,W_ = video_cthw.shape

        assert T_ == self.target_frames, "mismatched in T"
        assert H_ == self.target_h, "mismatched in H"
        assert W_ == self.target_w, "mismatched in W"

        # Handle Textual
        if len(text) >= self.T5_MAX_LEN * 2:
            text = text[:self.T5_MAX_LEN * 2]

        return video_cthw, text

    def __getitem__(self, index):
        for _ in range(self.tries):
            try:
                return self.getitem(index)
            except Exception as e:
                print(f"Error in loading one sample: {e}")
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.samples)
    
    def get_caption(self, index):
        pass

    @staticmethod
    def get_task_n_episode_id(sample_dir_name):
        assert len(sample_dir_name) == 12, f"Sample name len error"
        parts = sample_dir_name.split('-')
        assert len(parts) == 3, f"Sample name format error"
        task_str, episode_str, segment_str = parts
        return parts

    def make_video_path(self, task_str, episode_str, segment_str):
        video_path = os.path.join(
            self.agibot_dataset_path,
            f"{task_str}-{episode_str}-{segment_str}"
        )
        return video_path
    
    def get_video_path(self, sample_dir_name):
        video_path = os.path.join(
            self.agibot_dataset_path,
            sample_dir_name
        )
        return video_path

    def get_global_raw_frame_index(self, task_str, episode_str, target_segment_str):
        global_frame_start = 0
        for seg in range(int(target_segment_str)):
            info_dict = get_video_info(self.make_video_path(task_str, episode_str, seg))
            global_frame_start += info_dict['frames']
        info_dict_ = get_video_info(self.make_video_path(task_str, episode_str, target_segment_str))
        global_frame_end = info_dict_['frames']
        
        # trimed to reduce some marginal bug
        global_frame_start += 1
        global_frame_end -= 1

        return global_frame_start, global_frame_end



if __name__ == '__main__':

    # data_path = '/mnt/data-oss/openvid_tiny/data/train/mini_openvid.csv'
    # root='/mnt/data-oss/openvid_tiny/video'
    data_path = '/mnt/data-oss/openvid/data/train/OpenVid-1M.csv'
    root='/mnt/data-oss/openvid/video'

    dataset = AgibotDataset(
        data_path,
        target_frames=81,
        target_h=480,
        target_w=832,
        agibot_dataset_path=root,
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
    cnt = 0
    for video_data in loader:
        cnt += 1
        video_list, text_list = video_data
        if cnt == 10:
            print(video_list[0].shape)
            print(text_list[0])
            cthw_to_video(video_list[0],"meow.mp4")