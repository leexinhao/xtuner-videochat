srun -p videop1 \
    -N8 -n8 \
    --job-name=VC3_S4OL_v1 \
    --quotatype=reserved \
    --ntasks-per-node=1 \
    --cpus-per-task=128 \
    --gres=gpu:8 \
    -x SH-IDC1-10-140-37-[5-6,8,14,24,31,42,50] \
    --preempt \
    bash training_scripts/stage4_online/VideoChat3_4B_train_stage4_online_v1.sh

        # --preempt \