# from_path=videogpu:zhuyuhan/videochat3/checkpoints/stage1-2/VideoChat3_4B_train_stage1-2_sohav2
# to_path=/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/work_dir/stage1-2/VideoChat3_4B_train_stage1-2_sohav2

# rclone copy $from_path $to_path --transfers 16 --checkers 16 --progress

# echo "Copy $from_path to $to_path"



# to_path=/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/VC3_Online_data_annotations
# from_path=videogpu:zhuyuhan/videochat3/VC3_Online_data_annotations

# rclone copy $from_path $to_path --transfers 16 --checkers 16 --progress

# echo "Copy $from_path to $to_path"






# from_path=/mnt/petrelfs/zengxiangyu/Research_Zhang/eval_way/VLMEvalKit
# to_path=videogpu:zengxiangyu/VLMEvalKit

# rclone copy $from_path $to_path --transfers 32 --checkers 32 --progress

# echo "Copy $from_path to $to_path"




to_path=/mnt/petrelfs/zengxiangyu/Research_lixinhao/videochat3_data_annotations/video/timelines20260607/
from_path=videogpu:zhuyuhan/videochat3/videochat3_data_annotations/LV-Video/output_xtuner_format/timelines_merge_all_20260607_over10min_xtuner_format.jsonl

rclone copy $from_path $to_path --transfers 16 --checkers 16 --progress

echo "Copy $from_path to $to_path"



# to_path=/mnt/petrelfs/zengxiangyu/Research_lixinhao/Synthetic-Video/generated_300_per_category/
# from_path=videogpu:zhuyuhan/videochat3/videochat3_data_annotations/Synthetic-Video/generated_300_per_category/

# rclone copy $from_path $to_path --transfers 8 --checkers 8 --progress

# echo "Copy $from_path to $to_path"