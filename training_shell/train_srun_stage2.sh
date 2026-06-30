JOB_NAME=S2New_0609
NNODES=16


srun -p videop1 \
    -N${NNODES} -n${NNODES} \
    --job-name=${JOB_NAME} \
    --ntasks-per-node=1 \
    --cpus-per-task=128 \
    --gres=gpu:8 \
    --preempt \
    -x SH-IDC1-10-140-37-[7,50,55] \
    bash training_scripts/stage2_new/VideoChat3_4B_train_stage2_minisft_final_lr8e-5_vtlr2e-5_sohav2_0609.sh