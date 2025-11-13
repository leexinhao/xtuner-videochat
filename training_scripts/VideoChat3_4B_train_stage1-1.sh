set -ex
export PYTHONPATH="$(pwd)"
# export XTUNER_TOKENIZE_WORKERS=16
# export XTUNER_USE_FA3=1
current_time=$(date "+%m%d%H")
TASK_NAME="VideoChat3_4B_train_stage1-1"
OUTPUT_DIR="work_dir/${TASK_NAME}"
if [ ! -d "$OUTPUT_DIR" ]; then  
  mkdir -p "$OUTPUT_DIR"
fi
# -m debugpy --connect 5680
srun -p videoop --gres=gpu:8 --quotatype=spot \
torchrun --nproc-per-node=8 xtuner/v1/train/cli/sft.py --config training_configs/VideoChat3_4B_train_stage1-1.py 2>&1 | tee -a "${OUTPUT_DIR}/training_log_${TASK_NAME}_${current_time}.txt"