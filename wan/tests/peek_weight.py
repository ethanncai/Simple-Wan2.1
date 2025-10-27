import torch

ckpt_path = "/home/rapverse/workspace_junzhi/Wan2.1/exp/exp_ds_10271814/ds_step_100/global_step_100/mp_rank_00_model_states.pt"
state_dict = torch.load(ckpt_path, map_location="cpu")

model_state = state_dict.get('module', {})

found = False
for key in model_state.keys():
    if 'alpha' in key:  # 精确子串匹配（区分大小写）
        print(f"\n=== Key: {key} ===")
        print(model_state[key])
        found = True

if not found:
    print("No keys containing 'alpha' found in model state ('module').")
    print("Sample keys (first 20):", list(model_state.keys())[:20])