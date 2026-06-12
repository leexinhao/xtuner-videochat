PARTITION='videop1'
JOB_NAME='salloc1'
NNODE=1
NUM_GPUS=8
NUM_CPUS=128

salloc -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    -n${NNODE} \
    --gres=gpu:${NUM_GPUS} \
    --ntasks-per-node=1 \
    -w SH-IDC1-10-140-37-7 \
    --preempt \
    --cpus-per-task=${NUM_CPUS} \

    # -x SH-IDC1-10-140-37-[73,82,100,128,160]