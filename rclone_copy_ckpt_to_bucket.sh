# ckpt_name=VideoChat3_4B_train_stage4_online_v8
# ckpt_name=VideoChat3_4B_train_v11_online_v5
# ckpt_name=VideoChat3_4B_train_s3_v11_ol_v7
ckpt_name=VideoChat3_4B_train_s3_v11_ol_v9

ckpt_path=/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/work_dir/stage3_online/$ckpt_name
bucket_path=videogpu:zhuyuhan/videochat3/checkpoints/stage3_online/$ckpt_name

rclone copy $ckpt_path $bucket_path --transfers 16 --checkers 16 --progress

echo "Copy $ckpt_name to $bucket_path"