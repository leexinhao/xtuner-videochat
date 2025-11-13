import json

with open("/mnt/petrelfs/zengxiangyu/Research_lixinhao/vflash_annotations/caption_sharegpt4o-sharegpt4o_3k.json", "r") as f:
    for line in f:
        print(json.loads(line))
        break