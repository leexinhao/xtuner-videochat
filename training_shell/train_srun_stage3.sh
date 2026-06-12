JOB_NAME=S3_96k_Point
NNODES=16

srun -p videop1 \
    -N${NNODES} -n${NNODES} \
    --job-name=${JOB_NAME} \
    --ntasks-per-node=1 \
    --cpus-per-task=128 \
    --gres=gpu:8 \
    --preempt \
    bash training_scripts/stage3/VideoChat3_4B_train_stage3_minisft_finalbase_lr2e-5_long_96k_v11_Point.sh
