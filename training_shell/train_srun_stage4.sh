JOB_NAME=S4_128k_ol_SFv2_Streamo_Seeker
NNODES=16


srun -p videop1 \
    -N${NNODES} -n${NNODES} \
    --job-name=${JOB_NAME} \
    --ntasks-per-node=1 \
    --cpus-per-task=128 \
    --gres=gpu:8 \
    --preempt \
    bash training_scripts/stage4/VideoChat3_4B_train_stage4_v11base128k_lr2e-5_seq128k_v11_SFv2_Streamo_Seeker.sh
