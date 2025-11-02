import os
import json
import numpy as np
import h5py
import torch
from scipy.spatial.transform import Rotation
import cv2
from einops import rearrange
import matplotlib.cm as cm
import imageio.v3 as iio  
# --- 必要常量 ---
EEF2CamLeft = [0, 0, -0.5236]
EEF2CamRight = [0, 0, 0.5236]
ColorMapLeft = cm.Greens
ColorMapRight = cm.Reds
ColorListLeft = [(0, 0, 255), (255, 255, 0), (0, 255, 255)]
ColorListRight = [(255, 0, 255), (255, 0, 0), (0, 255, 0)]
EndEffectorPts = [
    [0, 0, 0, 1],
    [0.1, 0, 0, 1],
    [0, 0, 0.1, 1],
    [0, 0, 0.1, 1]
]
Gripper2EEFCvt = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0.23],
    [0, 0, 0, 1]
]

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def get_actions(gripper, all_ends_p=None, all_ends_o=None):
    n = all_ends_p.shape[0]
    cvt_vis_l = Rotation.from_euler("xyz", np.array(EEF2CamLeft))
    cvt_vis_r = Rotation.from_euler("xyz", np.array(EEF2CamRight))
    all_left_quat = []
    all_right_quat = []

    for i in range(n):
        rot_l = Rotation.from_quat(all_ends_o[i, 0])
        rot_vis_l = rot_l * cvt_vis_l
        left_vis_quat = np.concatenate((all_ends_p[i, 0], rot_vis_l.as_quat()), axis=0)

        rot_r = Rotation.from_quat(all_ends_o[i, 1])
        rot_vis_r = rot_r * cvt_vis_r
        right_vis_quat = np.concatenate((all_ends_p[i, 1], rot_vis_r.as_quat()), axis=0)

        all_left_quat.append(left_vis_quat)
        all_right_quat.append(right_vis_quat)

    all_left_quat = np.stack(all_left_quat)
    all_right_quat = np.stack(all_right_quat)

    all_abs_actions = np.zeros([n, 16])
    for i in range(n):
        all_abs_actions[i, 0:7] = all_left_quat[i, :7]
        all_abs_actions[i, 7] = gripper[i, 0]
        all_abs_actions[i, 8:15] = all_right_quat[i, :7]
        all_abs_actions[i, 15] = gripper[i, 1]
    return all_abs_actions

def get_transformation_matrix_from_quat(xyz_quat):
    rot_quat = xyz_quat[:, 3:]
    rot_quat = rot_quat[:, [3, 0, 1, 2]]  # to wxyz
    rot = quaternion_to_matrix(rot_quat)
    trans = xyz_quat[:, :3]
    output = torch.eye(4).unsqueeze(0).repeat(xyz_quat.shape[0], 1, 1)
    output[:, :3, :3] = rot
    output[:, :3, 3] = trans
    return output

@torch.no_grad()
def get_traj(sample_size, action, w2c, c2w, intrinsic, radius=50):
    h, w = sample_size
    if isinstance(action, np.ndarray):
        action = torch.tensor(action, dtype=torch.float32)
    
    device = action.device
    ee_key_pts = torch.tensor(EndEffectorPts, dtype=torch.float32, device=device).view(1, 1, 4, 4).permute(0, 1, 3, 2)

    pose_l_mat = get_transformation_matrix_from_quat(action[:, 0:7]).unsqueeze(0)
    pose_r_mat = get_transformation_matrix_from_quat(action[:, 8:15]).unsqueeze(0)
    
    ee2cam_l = torch.matmul(w2c, pose_l_mat)
    ee2cam_r = torch.matmul(w2c, pose_r_mat)

    cvt_matrix = torch.tensor(Gripper2EEFCvt, dtype=torch.float32, device=device).view(1, 1, 4, 4)
    ee2cam_l = torch.matmul(ee2cam_l, cvt_matrix)
    ee2cam_r = torch.matmul(ee2cam_r, cvt_matrix)
    
    pts_l = torch.matmul(ee2cam_l, ee_key_pts)
    pts_r = torch.matmul(ee2cam_r, ee_key_pts)
    
    intrinsic = intrinsic.unsqueeze(1)
    uvs_l = torch.matmul(intrinsic, pts_l[:, :, :3, :])
    uvs_l = (uvs_l / pts_l[:, :, 2:3, :])[:, :, :2, :].permute(0, 1, 3, 2)
    uvs_r = torch.matmul(intrinsic, pts_r[:, :, :3, :])
    uvs_r = (uvs_r / pts_r[:, :, 2:3, :])[:, :, :2, :].permute(0, 1, 3, 2)

    uvs_l = uvs_l.cpu().numpy().astype(np.int32)
    uvs_r = uvs_r.cpu().numpy().astype(np.int32)

    all_img_list = []
    for iv in range(w2c.shape[0]):
        img_list = []
        for i in range(action.shape[0]):
            img = np.zeros((h, w, 3), dtype=np.uint8)

            normalized_value_l = action[i, 7].item() / 120.0
            normalized_value_r = action[i, 15].item() / 120.0
            color_l = ColorMapLeft(normalized_value_l)[:3]
            color_r = ColorMapRight(normalized_value_r)[:3]
            color_l = tuple(int(c * 255) for c in color_l)
            color_r = tuple(int(c * 255) for c in color_r)

            # Draw circles
            for points, color in zip([uvs_l[iv, i], uvs_r[iv, i]], [color_l, color_r]):
                base_x, base_y = points[0]  # now int32
                if 0 <= base_x < w and 0 <= base_y < h:
                    cv2.circle(img, (base_x, base_y), radius, color, -1)

            # Draw lines
            for points, color, colors in zip([uvs_l[iv, i], uvs_r[iv, i]], [color_l, color_r], [ColorListLeft, ColorListRight]):
                base_x, base_y = points[0]
                if 0 <= base_x < w and 0 <= base_y < h:
                    for j in range(1, len(points)):  # skip j=0
                        pt_x, pt_y = points[j]
                        cv2.line(img, (base_x, base_y), (pt_x, pt_y), colors[j - 1], 8)

            img_list.append(img / 255.0)
        img_list = np.stack(img_list, axis=0)
        all_img_list.append(img_list)

    all_img_list = np.stack(all_img_list, axis=0)
    all_img_list = rearrange(torch.tensor(all_img_list), "v t h w c -> c v t h w").float()
    return all_img_list

def get_action_with_vp(h5_path,in_json_path,ex_json_path, resolution=(480, 640), radius=50, output_vp=False,action_transform=None):
    """
    Generate raw absolute actions and Visual Prompt (trajectory) in CTHW format.

    Returns:
        raw_actions (np.ndarray): Shape (T, 16)
        vp_traj (torch.Tensor): Shape (C, T, H, W) = (3, T, 720, 1280)
    """
    h, w = resolution

    # 1. Load raw actions
    with h5py.File(h5_path, "r") as fid:
        gripper = np.array(fid["state/effector/position"], dtype=np.float32)
        ends_p = np.array(fid["state/end/position"], dtype=np.float32)
        ends_o = np.array(fid["state/end/orientation"], dtype=np.float32)
    raw_actions = get_actions(gripper, ends_p, ends_o)  # (T, 16)
    
    raw_actions = torch.tensor(raw_actions)
    transformed_action = action_transform(raw_actions)
    if not output_vp:
        return transformed_action, None

    # 2. Load camera params (use first frame extrinsic)
    with open(ex_json_path, "r") as f:
        ex_info = json.load(f)
        c2w_np = np.eye(4)
        c2w_np[:3, :3] = np.array(ex_info[0]["extrinsic"]["rotation_matrix"])
        c2w_np[:3, 3] = np.array(ex_info[0]["extrinsic"]["translation_vector"])
    c2w = torch.from_numpy(c2w_np).float().unsqueeze(0)  # (1, 4, 4)
    w2c = torch.linalg.inv(c2w)

    with open(in_json_path, "r") as f:
        in_info = json.load(f)["intrinsic"]
    intrinsic_np = np.eye(3)
    intrinsic_np[0, 0] = in_info["fx"]
    intrinsic_np[1, 1] = in_info["fy"]
    intrinsic_np[0, 2] = in_info["ppx"]
    intrinsic_np[1, 2] = in_info["ppy"]
    intrinsic = torch.from_numpy(intrinsic_np).float().unsqueeze(0)  # (1, 3, 3)

    # 3. Generate trajectory with get_traj → returns (C, V, T, H, W)
    traj_ctvhw = get_traj(
        sample_size=(h, w),
        action=raw_actions,
        w2c=w2c,
        c2w=c2w,
        intrinsic=intrinsic,
        radius=radius
    )  # shape: (3, 1, T, 720, 1280)

    # 4. Remove the view dimension (V=1) → (C, T, H, W)
    vp_traj = traj_ctvhw.squeeze(1)  # Now (3, T, 720, 1280)

    return transformed_action, vp_traj.clone().detach().to(dtype=torch.float)



import os
import torch
from tqdm import tqdm
import numpy as np

def approximate_stats_from_samples(sample_dirs, radius=50, sample_ratio=9.9, seed=42):
    """
    Approximate min, max, mean, std per channel across many samples.
    
    Args:
        sample_dirs: list of sample directory paths
        radius: argument passed to get_action_with_vp
        sample_ratio: fraction of samples to use (e.g., 0.1 for 10%)
        seed: random seed for reproducibility if sampling
    """
    if sample_ratio < 1.0:
        np.random.seed(seed)
        n_samples = int(len(sample_dirs) * sample_ratio)
        sample_dirs = np.random.choice(sample_dirs, size=n_samples, replace=False).tolist()
    
    # We'll initialize stats after seeing the first tensor
    initialized = False
    mins = None
    maxs = None
    count = 0  # total number of time steps processed
    mean = None
    M2 = None  # for variance (Welford's algorithm)

    for sample_dir in tqdm(sample_dirs, desc="Processing samples"):
        try:
            ex_json_path = os.path.join(sample_dir,
                                    "head_extrinsic_params_aligned.json")
            in_json_path = os.path.join(sample_dir,
                                        "head_intrinsic_params.json")
            h5_json_path = os.path.join(sample_dir,
                                        "proprio_stats.h5")
            actions_tensor, _ = get_action_with_vp(h5_json_path, in_json_path, ex_json_path, radius=50, output_vp=False, action_transform=action_relative_to_0)
            # Ensure shape is [T, C]
            if actions_tensor.dim() == 3 and actions_tensor.shape[-1] == 1:
                actions_tensor = actions_tensor.squeeze(-1)  # [T, C, 1] -> [T, C]
            elif actions_tensor.dim() != 2:
                raise ValueError(f"Unexpected tensor shape: {actions_tensor.shape}")
            
            T, C = actions_tensor.shape

            if not initialized:
                # Initialize accumulators
                mins = torch.full((C,), float('inf'))
                maxs = torch.full((C,), float('-inf'))
                mean = torch.zeros(C)
                M2 = torch.zeros(C)
                initialized = True

            # Update min/max
            batch_min = actions_tensor.min(dim=0).values
            batch_max = actions_tensor.max(dim=0).values
            mins = torch.minimum(mins, batch_min)
            maxs = torch.maximum(maxs, batch_max)

            # Welford's online algorithm for mean and variance
            for t in range(T):
                x = actions_tensor[t]  # shape [C]
                count += 1
                delta = x - mean
                mean += delta / count
                delta2 = x - mean
                M2 += delta * delta2

        except Exception as e:
            print(f"Skipping {sample_dir}: {e}")
            continue

    if not initialized:
        raise ValueError("No valid samples processed.")

    # Compute std (unbiased estimator: divide by (count - 1))
    variance = M2 / (count - 1) if count > 1 else M2
    stds = torch.sqrt(torch.clamp(variance, min=0))

    # Print results
    for c in range(C):
        print(f"Channel {c}: min={mins[c]:.4f}, max={maxs[c]:.4f}, "
              f"mean={mean[c]:.4f}, std={stds[c]:.4f}")

    return {
        'min': mins,
        'max': maxs,
        'mean': mean,
        'std': stds,
        'total_timesteps': count,
        'num_samples_used': len(sample_dirs)
    }

import torch

ACTION_MEAN = torch.tensor([
    0.0186,   # Channel 0
    -0.0006,  # Channel 1
    -0.0061,  # Channel 2
    -0.0058,  # Channel 3
    0.0039,   # Channel 4
    0.0322,   # Channel 5
    0.0368,   # Channel 6
    7.8852,   # Channel 7
    0.0144,   # Channel 8
    0.0109,   # Channel 9
    -0.0215,  # Channel 10
    0.1183,   # Channel 11
    -0.0441,  # Channel 12
    -0.0006,  # Channel 13
    -0.0855,  # Channel 14
    10.8845,  # Channel 15
], dtype=torch.float32)

ACTION_STD = torch.tensor([
    0.0708,   # Channel 0
    0.0514,   # Channel 1
    0.0490,   # Channel 2
    0.2533,   # Channel 3
    0.2302,   # Channel 4
    0.2535,   # Channel 5
    0.1583,   # Channel 6
    31.6437,  # Channel 7
    0.0590,   # Channel 8
    0.0791,   # Channel 9
    0.0620,   # Channel 10
    0.5519,   # Channel 11
    0.4618,   # Channel 12
    0.1966,   # Channel 13
    0.4032,   # Channel 14
    37.0041,  # Channel 15
], dtype=torch.float32)

def scale_action(action: torch.Tensor) -> torch.Tensor:
    """
    Normalize action to zero-mean, unit-variance (Z-score).
    
    Args:
        action: Tensor of shape [..., C] or [..., C, 1]
    
    Returns:
        normalized_action: same shape as input
    """
    # Handle [..., C, 1] -> squeeze last dim if needed
    squeezed = False
    if action.shape[-1] == 1:
        action = action.squeeze(-1)
        squeezed = True

    # Ensure mean/std are on the same device and dtype
    mean = ACTION_MEAN.to(action.device, dtype=action.dtype)
    std = ACTION_STD.to(action.device, dtype=action.dtype)

    normalized = (action - mean) / std

    # Restore [..., C, 1] if input had it
    if squeezed:
        normalized = normalized.unsqueeze(-1)

    return normalized


def unscale_action(normalized_action: torch.Tensor) -> torch.Tensor:
    """
    Inverse of scale_action: recover original action values.
    
    Args:
        normalized_action: Tensor of shape [..., C] or [..., C, 1]
    
    Returns:
        original_action: same shape as input
    """
    squeezed = False
    if normalized_action.shape[-1] == 1:
        normalized_action = normalized_action.squeeze(-1)
        squeezed = True

    mean = ACTION_MEAN.to(normalized_action.device, dtype=normalized_action.dtype)
    std = ACTION_STD.to(normalized_action.device, dtype=normalized_action.dtype)
    original = normalized_action * std + mean

    if squeezed:
        original = original.unsqueeze(-1)

    return original


def action_relative_to_0(action: torch.Tensor) -> torch.Tensor:
    if action.ndim != 2:
        raise ValueError(f"Expected [T, dim], got {action.shape}")
    if action.shape[0] == 0:
        return action
    return action - action[0:1]  # broadcasting: [T, dim] - [1, dim]


# if __name__ == "__main__":
#     base_dir = "/home/rapverse/workspace_junzhi/datasets_ckpts/train"
#     sample_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir)
#                    if os.path.isdir(os.path.join(base_dir, d))]

#     # Optional: shuffle or sort if needed
#     # sample_dirs = sorted(sample_dirs)

#     # Run approximation (e.g., use 10% of data for speed)
#     stats = approximate_stats_from_samples(
#         sample_dirs,
#         radius=50,
#         sample_ratio=0.9,  # change to 1.0 for full dataset (slow!)
#         seed=42
#     )


if __name__ == "__main__":
    sample_dir_path = "/home/rapverse/workspace_junzhi/datasets_ckpts/train/574-840446-005"
    ex_json_path = os.path.join(sample_dir_path,
                                "head_extrinsic_params_aligned.json")
    in_json_path = os.path.join(sample_dir_path,
                                "head_intrinsic_params.json")
    h5_json_path = os.path.join(sample_dir_path,
                                "proprio_stats.h5")


    actions_tensor, _ = get_action_with_vp(h5_json_path, in_json_path, ex_json_path, radius=50, output_vp=False, action_transform=action_relative_to_0)

    actions_tensor_ = actions_tensor.clone()
    print(actions_tensor_.shape)
    # actions_tensor.unsqueeze(-1)
    mins = actions_tensor.min(dim=0).values
    maxs = actions_tensor.max(dim=0).values
    means = actions_tensor.mean(dim=0)
    stds = actions_tensor.std(dim=0)

    num_channels = actions_tensor.shape[1]
    for c in range(num_channels):
        print(f"Channel {c}: min={mins[c]:.4f}, max={maxs[c]:.4f}, "
              f"mean={means[c]:.4f}, std={stds[c]:.4f}")
        

    print(actions_tensor[0])
    print("doing scale") 
    
    actions_tensor = actions_tensor_

    actions_tensor = scale_action(actions_tensor)
    mins = actions_tensor.min(dim=0).values
    maxs = actions_tensor.max(dim=0).values
    means = actions_tensor.mean(dim=0)
    stds = actions_tensor.std(dim=0)

    num_channels = actions_tensor.shape[1]
    for c in range(num_channels):
        print(f"Channel {c}: min={mins[c]:.4f}, max={maxs[c]:.4f}, "
              f"mean={means[c]:.4f}, std={stds[c]:.4f}")
    # doing scale 


