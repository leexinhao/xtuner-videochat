#!/usr/bin/env sh

# 需要先salloc -p videoop -N4 -n4 --job-name=debug --ntasks-per-node=1 --cpus-per-task=128 --gres=gpu:8
set -ex
nnodes=4
master_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
all_nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
echo "All nodes used: ${all_nodes}"
echo "Master node ${master_node}"

head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$master_node" hostname --ip-address | awk '{print $1}')
# head_node_ip=$master_node
rdzv_endpoint="${head_node_ip}:${MASTER_PORT:-40000}"
bin="srun"

# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5
export XTUNER_USE_FA3="0"
# export XTUNER_PACK_WORKERS=8
# export XTUNER_TOKENIZE_WORKERS=16
export XTUNER_GC_ENABLE="1"

current_time=$(date "+%m%d%H%M%S")
TASK_NAME="VideoChat3_4B_train_stage1-2_new_cc3m"
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
$bin torchrun --nnodes=$nnodes \
        --nproc_per_node=8 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=${rdzv_endpoint} \
        xtuner/v1/train/cli/sft.py --config training_configs/videochat3/VideoChat3_4B_train_stage1-2_new_cc3m.py 2>&1 | tee -a "${OUTPUT_DIR}/training_log_${TASK_NAME}_${current_time}.txt"

