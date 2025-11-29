#!/usr/bin/env sh

# 使用srun -p videoop -N4 -n4 --job-name=stage1-2 --ntasks-per-node=1 --cpus-per-task=128 --gres=gpu:8
set -ex
cd /mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat

current_time=$(date "+%m%d%H")
TASK_NAME="VideoChat3_4B_train_stage1-2_debug_zyh"
OUTPUT_DIR="work_dir/${TASK_NAME}"
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
# torchrun --nnodes=$nnodes \
#         --nproc_per_node=8 \
#         --rdzv_backend=c10d \
#         --rdzv_endpoint=${rdzv_endpoint} \
#         xtuner/v1/train/cli/sft.py --config training_configs/videochat3/VideoChat3_4B_train_stage1-2.py 2>&1 | tee -a "${OUTPUT_DIR}/training_log_${TASK_NAME}_${current_time}.txt"

srun -p videoop --gres=gpu:8 --quotatype=spot \
torchrun --nproc_per_node=8 xtuner/v1/train/cli/sft.py \
  --config training_configs/debug/VideoChat3_4B_train_stage1-2_zyh.py 2>&1 | tee -a "${OUTPUT_DIR}/training_log_${TASK_NAME}_${current_time}.txt"