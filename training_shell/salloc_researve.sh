PARTITION='videop1'
JOB_NAME='online_debug'
NNODE=1
NUM_GPUS=8
NUM_CPUS=128

salloc -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    -n${NNODE} \
    --gres=gpu:${NUM_GPUS} \
    --ntasks-per-node=1 \
    --cpus-per-task=${NUM_CPUS} \
    --preempt
    
    # -w SH-IDC1-10-140-37-75 \
    # --preempt \
