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

tokenize_fn = VideoChat3TokenizeFnConfig(processor_path=VIDEOCHAT3_PATH,
                                        add_vision_id=add_vision_id).build(self_tokenizer)
data_path = '/mnt/petrelfs/zengxiangyu/Research_lixinhao/videochat3_data_annoations/video/caption_smit_481k.jsonl'
with open(data_path, encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 100:
            break
        raw_data = json.loads(line)

        # ret_xtuner = tokenize_fn(raw_data, media_root=LOCAL_MEDIA_ROOT)
        # input_ids_xtuner = ret_xtuner['input_ids']
        # pixel_values_xtuner: torch.Tensor = ret_xtuner['pixel_values']
        # video_grid_thw_xtuner: torch.Tensor = ret_xtuner['image_grid_thw']

        ret_xtuner_ceph = tokenize_fn(raw_data, media_root=CEPH_ROOT)
        input_ids_xtuner_ceph = ret_xtuner_ceph['input_ids']
        pixel_values_xtuner_ceph: torch.Tensor = ret_xtuner_ceph['pixel_values']
        video_grid_thw_xtuner_ceph: torch.Tensor = ret_xtuner_ceph['image_grid_thw']
        
        raw_data_copy = json.loads(line)
        tokenize_fn.state = "cache"
        cache_result = tokenize_fn(raw_data_copy, media_root=LOCAL_MEDIA_ROOT)
        tokenize_fn.state = "get_item"
        assert len(input_ids_xtuner_ceph) == cache_result['num_tokens'], f"calc_num_tokens_get_item{cache_result['num_tokens']}和get_item出来的token数{len(input_ids_xtuner_ceph)}不一致！"
        print(i, f"通过！num_tokens={cache_result['num_tokens']}", flush=True)
    