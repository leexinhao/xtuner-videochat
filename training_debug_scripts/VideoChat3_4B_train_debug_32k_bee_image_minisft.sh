#!/usr/bin/env sh

# 需要先salloc -p videoop -N4 -n4 --job-name=debug --ntasks-per-node=1 --cpus-per-task=128 --gres=gpu:8
set -ex

# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5
export XTUNER_USE_FA3="0"
# export XTUNER_PACK_WORKERS=8
# export XTUNER_TOKENIZE_WORKERS=16
export XTUNER_GC_ENABLE="1"
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

current_time=$(date "+%m%d%H%M%S")
TASK_NAME="VideoChat3_4B_train_debug_32k_bee_image_minisft"
OUTPUT_DIR="work_dir/stage1-2_debug/${TASK_NAME}"
if [ ! -d "$OUTPUT_DIR" ]; then  
  mkdir -p "$OUTPUT_DIR"
fi

echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

#run command
srun -p videoop --gres=gpu:8 --quotatype=spot \
torchrun --nproc-per-node=8 xtuner/v1/train/cli/sft.py --config training_configs/videochat3_debug/${TASK_NAME}.py 2>&1 | tee -a "${OUTPUT_DIR}/training_log_${TASK_NAME}_${current_time}.txt"

