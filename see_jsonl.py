import json

with open("/mnt/petrelfs/zengxiangyu/Research_lixinhao/videochat3_data_annoations/image/DenseFusion_xtuner_qwen3_recap_clean_repeat_2round.jsonl", "r") as fr:
    for line in fr:
        print(json.loads(line))
        break