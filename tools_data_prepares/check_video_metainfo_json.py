import os
from unittest import TestCase
import torch
from xtuner.v1.datasets import VideoChat3TokenizeFnConfig
from xtuner.v1.datasets.mllm_tokenize_fn.base_mllm_tokenize_fn import collect_image_video_paths_and_extra
from transformers import AutoTokenizer, AutoProcessor
import json
import parametrize

LOCAL_MEDIA_ROOT = "tests/resource"

VIDEOCHAT3_PATH = "/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/VideoChat3-4B"
add_vision_id = False

self_tokenizer = AutoTokenizer.from_pretrained(VIDEOCHAT3_PATH, trust_remote_code=True)
# self.tokenize_fn = VideoChat3TokenizeFnConfig(processor_path=VIDEOCHAT3_PATH).build(self.tokenizer)
# self.processor = AutoProcessor.from_pretrained(VIDEOCHAT3_PATH, trust_remote_code=True)


with open('/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/stage1-2_final_debug1.json', 'r') as f:
    infos = json.load(f)
    
    for k in infos.keys():
        item = infos[k]
        print(f"开始检查{item}数据集")
        data_path = item['anno_path']
        CEPH_ROOT = item['media_root']
        sample_max_length = 8192 * 2
        tokenize_fn = VideoChat3TokenizeFnConfig(
                    max_length=sample_max_length,
                    image_min_pixels=item.get('image_min_pixels', 28*28),
                    image_max_pixels=item.get('image_max_pixels', int(sample_max_length * 0.7 * 28 * 28)),
                    frame_min_pixels=item.get('frame_min_pixels', 28*28),
                    frame_max_pixels=item.get('frame_max_pixels', int(640*480)),
                    video_max_total_pixels=item.get('video_max_total_pixels', int(sample_max_length * 0.6 * 4 * 28 * 28)),
                    video_min_frames=item.get('video_min_frames', 1),
                    video_max_frames=item.get('video_max_frames', 256), 
                    fixed_num_sampled_frames=item.get('fixed_num_sampled_frames', None),
                    video_sample_fps=item.get('video_sample_fps', 2), 
                    processor_path=VIDEOCHAT3_PATH,
                    data_augment=item.get('data_augment', False),
                    system_message=item.get('system_message', None),
                    hash=item.get('hash', None),
                    ).build(self_tokenizer)

        with open(data_path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i > 100:
                    break
                raw_data = json.loads(line)
                tokenize_fn.state = "get_item"
                # ret_xtuner = tokenize_fn(raw_data, media_root=LOCAL_MEDIA_ROOT)
                ret_xtuner = tokenize_fn(raw_data, media_root=CEPH_ROOT)
                input_ids_xtuner = ret_xtuner['input_ids']
                pixel_values_xtuner: torch.Tensor = ret_xtuner['pixel_values']
                video_grid_thw_xtuner: torch.Tensor = ret_xtuner['image_grid_thw']
                
                raw_data_copy = json.loads(line)
                tokenize_fn.state = "cache"
                cache_result = tokenize_fn(raw_data_copy, media_root=CEPH_ROOT)

                if len(input_ids_xtuner) != cache_result['num_tokens']:
                    print(raw_data)
                    print(i, f"calc_num_tokens_get_item{cache_result['num_tokens']}和get_item出来的token数{len(input_ids_xtuner)}不一致！", flush=True)
                    break
                else:
                    print(i, f"通过！num_tokens={cache_result['num_tokens']}", flush=True)
        