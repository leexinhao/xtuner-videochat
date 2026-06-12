PARTITION='video5'
JOB_NAME='spot2'
NNODE=4
NUM_GPUS=8
NUM_CPUS=128

salloc -p ${PARTITION} \
    --quotatype=spot \
    --job-name=${JOB_NAME} \
    -n${NNODE} \
    --gres=gpu:${NUM_GPUS} \
    --ntasks-per-node=1 \
    --cpus-per-task=${NUM_CPUS} \

    # -x SH-IDC1-10-140-37-[73,82,100,128,160]