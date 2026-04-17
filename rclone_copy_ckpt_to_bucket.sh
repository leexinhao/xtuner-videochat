ckpt_name=VideoChat3_4B_train_stage4_online_v3

ckpt_path=/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/work_dir/stage4_online/$ckpt_name
bucket_path=videogpu:zhuyuhan/videochat3/checkpoints/stage4_online/$ckpt_name

rclone copy $ckpt_path $bucket_path --transfers 16 --checkers 16 --progress

echo "Copy $ckpt_name to $bucket_path"