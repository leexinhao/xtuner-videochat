import json

compress_rate = 4
patch_size = 14

total_num_tokens = 0
with open("/mnt/petrelfs/zengxiangyu/Research_lixinhao/vflash_annotations/image_metainfos/blip_laion_cc_sbu_558k_metainfos.jsonl", "r") as f:
    for line in f:
        info = json.loads(line)
        num_tokens = (info['width'] * info['height']) // (compress_rate * patch_size * patch_size)
        # print(num_tokens)
        total_num_tokens += num_tokens

print(total_num_tokens)