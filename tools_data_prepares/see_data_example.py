import os
from unittest import TestCase
import torch
import json
import parametrize


data_meta_path = "/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/training_data_annotations/data_stage1-2_video_only.json"
with open(data_meta_path, "r") as fr:
    data_metas = json.load(fr)

for data_name in data_metas.keys():
    print("Checking:", data_name)
    data_path = data_metas[data_name]["anno_path"]
    total_tokens = 0
    with open(data_path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            raw_data = json.loads(line)
            print(raw_data)
            break
[]