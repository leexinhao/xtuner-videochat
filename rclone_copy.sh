from_path=videogpu:zhuyuhan/videochat3/checkpoints/stage1-2/VideoChat3_4B_train_stage1-2_sohav2
to_path=/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/work_dir/stage1-2/VideoChat3_4B_train_stage1-2_sohav2

rclone copy $from_path $to_path --transfers 16 --checkers 16 --progress

echo "Copy $from_path to $to_path"