import json

compress_rate = 4
patch_size = 14

total_num_tokens = 0
max_width = 0
max_height = 0
max_image = None
with open("/mnt/petrelfs/zengxiangyu/Research_lixinhao/vflash_annotations/image_metainfos/blip_laion_cc_sbu_558k_metainfos.jsonl", "r") as f:
    for line in f:
        info = json.loads(line)
        # num_tokens = (info['width'] * info['height']) // (compress_rate * patch_size * patch_size)
        # # print(num_tokens)
        # total_num_tokens += num_tokens
        if info['width'] > 10000 or info['height'] > 10000:
            print(info)
        # max_height = max(info['height'], max_height)

# print(max_width, max_height)