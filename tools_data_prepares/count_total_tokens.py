import os
from unittest import TestCase
import torch
from xtuner.v1.datasets import VideoChat3TokenizeFnConfig
from xtuner.v1.datasets.mllm_tokenize_fn.base_mllm_tokenize_fn import collect_image_video_paths_and_extra
from transformers import AutoTokenizer, AutoProcessor
import json
import parametrize

LOCAL_MEDIA_ROOT = "tests/resource"
CEPH_ROOT = "pvideo:s3://S-MiT/"
VIDEOCHAT3_PATH = os.environ.get("VIDEOCHAT3_PATH", "VideoChat3-2B")
add_vision_id = False

self_tokenizer = AutoTokenizer.from_pretrained(VIDEOCHAT3_PATH, trust_remote_code=True)
# self.tokenize_fn = VideoChat3TokenizeFnConfig(processor_path=VIDEOCHAT3_PATH).build(self.tokenizer)
# self.processor = AutoProcessor.from_pretrained(VIDEOCHAT3_PATH, trust_remote_code=True)
sample_max_length = 8192
tokenize_fn = VideoChat3TokenizeFnConfig(
                    max_length=sample_max_length,
                    image_min_pixels=28*28,
                    image_max_pixels=int(sample_max_length * 0.8 * 28 * 28),
                    frame_min_pixels=28*28,
                    frame_max_pixels=int(sample_max_length * 0.8 * 28 * 28),
                    video_max_total_pixels= int(sample_max_length * 0.8 * 4 * 28 * 28),
                    video_min_frames=1,
                    video_max_frames=2048, 
                    fixed_num_sampled_frames=None,
                    video_sample_fps=4, 
                    processor_path=VIDEOCHAT3_PATH,
                    # data_augment=_data.get('data_augment', False),
                    # system_message=_data.get('system_message', None),
                    # hash=_data.get('hash', None),
                    ).build(self_tokenizer)

print(tokenize_fn._hash_str)
data_meta_path = "/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/training_data_annotations/data_stage1-2_video_only.json"
with open(data_meta_path, "r") as fr:
    data_metas = json.load(fr)

token_dict = {"tokenize_hash_str": tokenize_fn._hash_str}

for data_name in data_metas.keys():
    print("Checking:", data_name)
    data_path = data_metas[data_name]["anno_path"]
    total_tokens = 0
    with open(data_path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            raw_data = json.loads(line)
            tokenize_fn.state = "cache"
            # ret_xtuner = tokenize_fn(raw_data, media_root=LOCAL_MEDIA_ROOT)
            cache_result = tokenize_fn(raw_data, media_root=CEPH_ROOT)
            # cache_result = tokenize_fn(raw_data, media_root=LOCAL_MEDIA_ROOT)
            total_tokens += cache_result['num_tokens']
            
            print(i, f"通过！num_tokens={cache_result['num_tokens']}", flush=True)
    token_dict[data_name] = total_tokens

with open(f"/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/tools_data_prepares/token_infos/{data_meta_path.split('/')[-1]}", "w") as fw:
    json.dump(token_dict, fw)