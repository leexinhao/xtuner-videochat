ckpt_name=VideoChat3_4B_train_stage4_v11base128k_lr2e-5_seq128k_v11_SFv2_Streamo_Seeker


ckpt_path=/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/work_dir/stage4/$ckpt_name
bucket_path=videogpu:zhuyuhan/videochat3/checkpoints/stage4/$ckpt_name

rclone copy $ckpt_path $bucket_path --transfers 16 --checkers 16 --progress

echo "Copy $ckpt_name to $bucket_path"









# ckpt_name=VideoChat3_4B_train_stage4_online_v11base_lr2e-5_v11_OL_v3


# local_path=/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/work_dir/stage4_online/$ckpt_name
# remote_path=videogpu:zhuyuhan/videochat3/checkpoints/stage4_online/$ckpt_name

# rclone copy $remote_path $local_path --transfers 16 --checkers 16 --progress

# echo "Copy $ckpt_name to $local_path"







# ckpt_path=/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/work_dir/stage2_new/
# bucket_path=videogpu:zhuyuhan/videochat3/checkpoints/stage2_new/

# rclone copy $ckpt_path $bucket_path --transfers 16 --checkers 16 --progress

# echo "Copy $ckpt_name to $bucket_path"