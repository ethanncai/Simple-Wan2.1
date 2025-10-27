#!/bin/bash
set -e  # 遇到错误立即退出

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8

export TORCH_DISTRIBUTED_DEBUG=OFF          # 关闭 PyTorch 分布式调试日志
export NCCL_DEBUG=warn                      # NCCL 仅显示警告（可选: silent）
export NCCL_ASYNC_ERROR_HANDLING=1          # 启用异步错误处理（避免卡死）
export PYTHONUNBUFFERED=1                   # 实时输出 Python 日志

SCRIPT="run_train.py"
DS_CONFIG="ds_config_zero2.json"

echo "开始训练..."
echo "使用 GPU: $CUDA_VISIBLE_DEVICES (共 $NUM_GPUS 卡)"
deepspeed \
    --num_gpus $NUM_GPUS \
    $SCRIPT \
    --ds_config $DS_CONFIG \
    2>&1 | tee training_$(date +%Y%m%d_%H%M%S).log

