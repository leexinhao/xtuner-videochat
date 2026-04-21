JOB_NAME=VC3_S4OL
NNODES=8


VERTION=v7

# VERTION=debug

srun -p videop1 \
    -N${NNODES} -n${NNODES} \
    --job-name=${JOB_NAME}_${VERTION} \
    --ntasks-per-node=1 \
    --cpus-per-task=128 \
    --gres=gpu:8 \
    --preempt \
    bash training_scripts/stage4_online/VideoChat3_4B_train_stage4_online_${VERTION}.sh
        # --preempt \