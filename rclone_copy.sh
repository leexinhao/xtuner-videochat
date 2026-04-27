from_path=videogpu:zhuyuhan/videochat3/videochat3_data_annotations/motion-videos/
to_path=/mnt/petrelfs/zengxiangyu/Research_lixinhao/videochat3_data_annotations/motion-videos/

rclone copy $from_path $to_path --transfers 16 --checkers 16 --progress

echo "Copy $from_path to $to_path"