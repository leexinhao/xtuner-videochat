import json
import os
with open("/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/training_data_annotations/data_stage2_llava_video.json", "r") as fr:
    infos = json.load(fr)

for k in infos:
    info = infos[k]
    assert os.path.exists(info['anno_path']), info['anno_path']

