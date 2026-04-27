JOB_NAME=V11_OL
NNODES=8


VERTION=v3

# VERTION=debug

srun -p videop1 \
    -N${NNODES} -n${NNODES} \
    --job-name=${JOB_NAME}_${VERTION} \
    --ntasks-per-node=1 \
    --cpus-per-task=128 \
    --gres=gpu:8 \
    --preempt \
    bash training_scripts/stage4_online/VideoChat3_4B_train_v11_online_${VERTION}.sh
        # --preempt \