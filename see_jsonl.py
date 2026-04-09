import json
import os
# print(len(os.listdir("/mnt/petrelfs/zengxiangyu/Research_lixinhao/videochat3_data_annotations/image/honey_meta_merged_no_think")))
# print(len(os.listdir("/mnt/petrelfs/zengxiangyu/Research_lixinhao/videochat3_data_annotations/image/honey_meta_merged_no_think_fix")))


with open("/mnt/petrelfs/zengxiangyu/Research_lixinhao/videochat3_data_annotations/lvdb_grounding/lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_long_vc3format.jsonl", "r") as fr:
    for line in fr:
        try:
            print(json.loads(line))
            break
        except json.JSONDecodeError as e:
            print(f"[WARN] {line} JSON解析失败: {e}")
            continue