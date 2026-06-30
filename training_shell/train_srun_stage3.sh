JOB_NAME=S3N_96k_v11_iv3
NNODES=16

srun -p videop1 \
    -N${NNODES} -n${NNODES} \
    --job-name=${JOB_NAME} \
    --ntasks-per-node=1 \
    --cpus-per-task=128 \
    --gres=gpu:8 \
    --preempt \
    -x SH-IDC1-10-140-37-[7,50,55] \
    bash training_scripts/stage3_new/VideoChat3_4B_train_stage3new_lr2e-5_96k_v11_internvideo3.sh
