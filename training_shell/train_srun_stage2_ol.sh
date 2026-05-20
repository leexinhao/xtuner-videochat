JOB_NAME=s2_suoha
NNODES=16


VERTION=ol_v1

# VERTION=debug

srun -p videop1 \
    -N${NNODES} -n${NNODES} \
    --job-name=${JOB_NAME}_${VERTION} \
    --ntasks-per-node=1 \
    --cpus-per-task=128 \
    --gres=gpu:8 \
    --preempt \
    -x SH-IDC1-10-140-37-12 \
    bash training_scripts/stage2_new/VideoChat3_4B_train_stage2_minisft_final_lr8e-5_vtlr2e-5_sohav2_${VERTION}.sh
        # --preempt \

# /mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/work_dir/stage2/VideoChat3_4B_train_stage2_image_video_minisft_final_lr8e-5_vtlr2e-5_sohav2/20260322024754/hf-6060