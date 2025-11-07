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
from ..utils.train_utils import video_to_cthw, cthw_to_video, resize_but_retain_ratio, get_video_info, stretchly_resize
import torch.nn.functional as F
from pathlib import Path
import pickle
from ..utils.action_utils import get_action_with_vp, action_relative_to_0, scale_action


def _get_cache_path(dataset_path):
    parent = os.path.dirname(os.path.abspath(dataset_path))
    cache_name = "agibot_samples_cache.pkl"
    return os.path.join(parent, cache_name)


def _load_or_build_samples(dataset_path, cache_path):
    current_mtime = os.path.getmtime(dataset_path)
    
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            if cached.get('dataset_path') == dataset_path and cached.get('mtime') == current_mtime:
                print("[Dataset] Loading samples from cache...")
                return cached['samples'], True
            else:
                print("[Dataset] Cache exists but dataset path or mtime mismatch, rebuilding cache.")
        except Exception as e:
            print(f"[Dataset] Failed to load cache: {e}, rebuilding cache.")
    else:
        print("[Dataset] No cache found, building samples list...")

    samples = list(os.listdir(dataset_path))
    cache_data = {
        'dataset_path': dataset_path,
        'mtime': current_mtime,
        'samples': samples
    }
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print("[Dataset] Cache saved.")
    except Exception as e:
        print(f"[Dataset] Failed to save cache: {e}")
    
    return samples, False


class AgibotDataset(torch.utils.data.Dataset):
    """load video according to the csv file.

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(
        self,
        target_w,
        target_h,
        target_frames=81,
        sample_interval=2,  # means slow down 6x
        tries=10000,
        agibot_dataset_path=None,
        agibot_task_info_path=None,
        return_vp=False,
    ):
        # 这个类只在加载的时候进行完整性检查。indexing的时候不对样本是否corrupt进行检查
        self.tries = tries
        self.target_frames = target_frames
        self.target_h = target_h
        self.target_w = target_w
        self.agibot_dataset_path = agibot_dataset_path
        self.agibot_task_info_path = agibot_task_info_path
        self.T5_MAX_LEN = 512
        self.sample_interval = sample_interval
        self.target_raw_frames = self.target_frames * self.sample_interval
        self.return_vp = return_vp

        print("[Dataset] Indexing Dataset...")
        cache_path = _get_cache_path(self.agibot_dataset_path)
        self.samples, used_cache = _load_or_build_samples(self.agibot_dataset_path, cache_path)

        print(f"[Dataset] Dataset size: {len(self.samples)}")

    def getitem(self, index):
        sample_dir_name = self.samples[index]
        sample_dir_path = os.path.join(self.agibot_dataset_path, sample_dir_name)
        task_str, episode_str, segment_str = self.get_task_n_episode_id(sample_dir_name)

        video_path = os.path.join(sample_dir_path, "head_color.mp4")
        ex_json_path = os.path.join(sample_dir_path, "head_extrinsic_params_aligned.json")
        in_json_path = os.path.join(sample_dir_path, "head_intrinsic_params.json")
        h5_json_path = os.path.join(sample_dir_path, "proprio_stats.h5")

        # load video
        video_cthw = video_to_cthw(video_path)
        _, T, H, W = video_cthw.shape

        # load action
        action_tdim, vp_cthw = get_action_with_vp(
            h5_json_path, in_json_path, ex_json_path, radius=50,
            output_vp=self.return_vp, action_transform=action_relative_to_0
        )
        action_tdim = scale_action(action_tdim)
        action_len = action_tdim.shape[0]

        assert T == action_len
        if self.return_vp:
            assert T == vp_cthw.shape[1]
        if T < self.target_raw_frames:
            # pad video
            pad = self.target_raw_frames - T
            video_cthw = torch.cat(
                [video_cthw, video_cthw[:, -1:].repeat_interleave(pad, dim=1)],
                dim=1
            )
            # pad action
            action_tdim = torch.cat(
                [action_tdim, action_tdim[-1:].repeat_interleave(pad, dim=0)],
                dim=0
            )
            if self.return_vp:
                vp_cthw = torch.cat(
                    [vp_cthw, vp_cthw[:, -1:].repeat_interleave(pad, dim=1)],
                    dim=1
                )

            T = self.target_raw_frames  # update the t

        # load caption
        start_idx = torch.randint(0, T - self.target_raw_frames + 1, (1,)).item()
        end_idx = start_idx + self.target_raw_frames

        # global_start index
        global_frame_start_of_this_seg = self.get_global_raw_frame_number_of_seg(
            task_str, episode_str, segment_str, video_path
        )
        global_start_idx = global_frame_start_of_this_seg + start_idx
        global_end_idx = global_frame_start_of_this_seg + end_idx

        # load caption
        caption = self.get_caption(task_str, episode_str, segment_str, video_path, global_start_idx, global_end_idx)

        # Handle Spatial
        video_cthw = stretchly_resize(video_cthw, (self.target_h, self.target_w))
        if self.return_vp:
            vp_cthw = stretchly_resize(vp_cthw, (self.target_h, self.target_w))

        # Handle Temporal
        video_cthw = video_cthw[:, start_idx:end_idx, ...]
        if self.return_vp:
            vp_cthw = vp_cthw[:, start_idx:end_idx, ...]
        action_tdim = action_tdim[start_idx:end_idx, ...]

        assert action_tdim.shape[0] == video_cthw.shape[1] and action_tdim.shape[0] == self.target_raw_frames
        if self.return_vp:
            assert action_tdim.shape[0] == vp_cthw.shape[1]

        # sample T dimension according to sample_interval
        indices = torch.arange(0, self.target_raw_frames, self.sample_interval)
        video_cthw = video_cthw[:, indices, :, :]
        action_tdim = action_tdim[indices, :]
        if self.return_vp:
            vp_cthw = vp_cthw[:, indices, :, :]

        assert action_tdim.shape[0] == video_cthw.shape[1] and action_tdim.shape[0] == self.target_frames
        if self.return_vp:
            assert action_tdim.shape[0] == vp_cthw.shape[1]

        # Handle Textual
        if len(caption) >= self.T5_MAX_LEN * 2:
            caption = caption[:self.T5_MAX_LEN * 2]

        return video_cthw, action_tdim, caption, vp_cthw

    def __getitem__(self, index):
        for _ in range(self.tries):
            try:
                return self.getitem(index)
            except Exception as e:
                print(f"Error in loading one sample: {e}")
                # raise
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.samples)

    def get_caption(self, task_str, episode_str, segment_str, video_path, global_idx_start, global_idx_end):
        task_caption_json = os.path.join(self.agibot_task_info_path, f"task_{task_str}.json", f"task_{task_str}.json")
        with open(task_caption_json, 'r', encoding='utf-8') as f:
            task_captions = json.load(f)

        target_episode_data = None
        for episode_data in task_captions:
            if int(episode_data["episode_id"]) == int(episode_str):
                target_episode_data = episode_data
                break
        assert target_episode_data is not None, "not task info found"
        return self.extract_action_descriptions(target_episode_data, video_path, global_idx_start, global_idx_end)

    @staticmethod
    def extract_action_descriptions(data, video_path, target_start, target_end):
        """
        根据目标帧范围，从 label_info['action_config'] 中提取最相关的动作描述。
        """
        actions = data["label_info"]["action_config"]

        target_end = int(actions[-1]["end_frame"]) if target_end > int(actions[-1]["end_frame"]) else target_end
        if not actions:
            raise ValueError("action_config is empty")

        target_range = (target_start, target_end)
        target_len = max(1, target_end - target_start)

        matches = []
        overlaps = []

        for action in actions:
            a_start = action["start_frame"]
            a_end = action["end_frame"]
            a_len = max(1, a_end - a_start)

            overlap_start = max(a_start, target_start)
            overlap_end = min(a_end, target_end)
            overlap_len = max(0, overlap_end - overlap_start)

            if a_start >= target_start and a_end <= target_end:
                matches.append(action["action_text"])
                overlaps.append((overlap_len, action))
                continue

            if overlap_len > 0:
                overlap_ratio_action = overlap_len / a_len
                overlap_ratio_target = overlap_len / target_len
                if overlap_ratio_action >= 0.5 or overlap_ratio_target >= 0.5:
                    matches.append(action["action_text"])
                    overlaps.append((overlap_len, action))
                    continue

            if overlap_len > 0:
                overlaps.append((overlap_len, action))

        if matches:
            return ", and ".join(matches)

        if overlaps:
            texts = [act["action_text"] for _, act in overlaps]
            return ", and ".join(texts)

        def distance_to_interval(a_start, a_end, t_start, t_end):
            if a_end <= t_start:
                return t_start - a_end
            elif a_start >= t_end:
                return a_start - t_end
            else:
                return 0

        closest_action = min(
            actions,
            key=lambda act: distance_to_interval(
                act["start_frame"], act["end_frame"], target_start, target_end
            )
        )
        return closest_action["action_text"]

    @staticmethod
    def get_task_n_episode_id(sample_dir_name):
        assert len(sample_dir_name) == 12 or len(sample_dir_name) == 14, f"Sample name len error: {sample_dir_name}"
        parts = sample_dir_name.split('-')
        assert len(parts) == 3, f"Sample name format error: {sample_dir_name}"
        return parts

    def get_sample_path(self, sample_dir_name):
        video_path = os.path.join(self.agibot_dataset_path, sample_dir_name)
        return video_path

    def get_global_raw_frame_number_of_seg(self, task_str, episode_str, target_segment_str, video_path):
        global_frame_start = 0
        for seg in range(int(target_segment_str)):
            info_dict = get_video_info(video_path)
            global_frame_start += info_dict['frames']
        global_frame_start += 1
        return global_frame_start


if __name__ == '__main__':

    agibot_dataset_path = '/home/rapverse/workspace_junzhi/datasets_ckpts/train'
    agi_taskinfo_path = '/mnt/data-oss/rap-prod-bak/AgibotWorld-Beta/task_info'

    dataset = AgibotDataset(
        target_frames=81,
        sample_interval=2,
        target_h=480,
        target_w=832,
        agibot_dataset_path=agibot_dataset_path,
        agibot_task_info_path=agi_taskinfo_path,
        return_vp=True
    )

    def collate_fn(batch):
        video_tensors = [item[0] for item in batch]
        actions = [item[1] for item in batch]
        texts = [item[2] for item in batch]
        vps = [item[3] for item in batch]
        return video_tensors, actions, texts, vps

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
        video_list, action_list, text_list, vp_list = video_data
        print(f"Step {cnt}")
        if cnt == 1:
            print(video_list[0].shape)
            print(text_list[0])
            print(action_list[0].shape)
            print(vp_list[0].shape)
            cthw_to_video(video_list[0], "meow.mp4", fps=16)
            cthw_to_video(vp_list[0], "mewo_vp.mp4", fps=16)
            break