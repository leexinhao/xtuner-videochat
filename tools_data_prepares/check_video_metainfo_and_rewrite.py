import os
from unittest import TestCase
import torch
from xtuner.v1.datasets import VideoChat3TokenizeFnConfig
from xtuner.v1.datasets.mllm_tokenize_fn.base_mllm_tokenize_fn import collect_image_video_paths_and_extra
from transformers import AutoTokenizer, AutoProcessor
import json
import parametrize

LOCAL_MEDIA_ROOT = "tests/resource"
CEPH_ROOT = ""
VIDEOCHAT3_PATH = "/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/work_dir/stage1-1/VideoChat3_4B_train_stage1-1_old/20251127192016/hf-190"
add_vision_id = False

self_tokenizer = AutoTokenizer.from_pretrained(VIDEOCHAT3_PATH, trust_remote_code=True)
# self.tokenize_fn = VideoChat3TokenizeFnConfig(processor_path=VIDEOCHAT3_PATH).build(self.tokenizer)
# self.processor = AutoProcessor.from_pretrained(VIDEOCHAT3_PATH, trust_remote_code=True)
sample_max_length = 8192 * 4
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
data_path = '/mnt/petrelfs/zengxiangyu/Research_lixinhao/videochat3_data_annotations/video/tarster_video_clean/lsmdc_p1_tarsier_recap_26544_clean.jsonl'
save_path = '/mnt/petrelfs/zengxiangyu/Research_lixinhao/videochat3_data_annotations/video/tarster_video_clean/lsmdc_p1_tarsier_recap_26544_clean2.jsonl'

wrong_cnt = 0
with open(data_path, encoding='utf-8') as f:
    with open(save_path, 'w') as f_w:
        for i, line in enumerate(f):
            raw_data = json.loads(line)
            tokenize_fn.state = "get_item"
            try:
                # ret_xtuner = tokenize_fn(raw_data, media_root=LOCAL_MEDIA_ROOT)
                ret_xtuner = tokenize_fn(raw_data, media_root=CEPH_ROOT)
            except Exception as e:
                wrong_cnt += 1
                raise
                print(i, f"get 不通过！{e}", flush=True)
                continue
            input_ids_xtuner = ret_xtuner['input_ids']
            pixel_values_xtuner: torch.Tensor = ret_xtuner['pixel_values']
            video_grid_thw_xtuner: torch.Tensor = ret_xtuner['image_grid_thw']
            
            raw_data_copy = json.loads(line)
            tokenize_fn.state = "cache"
            try:
                cache_result = tokenize_fn(raw_data_copy, media_root=CEPH_ROOT)
            except Exception as e:
                wrong_cnt += 1
                print(i, f"cache 不通过！{e}", flush=True)
                continue

            if len(input_ids_xtuner) == cache_result['num_tokens']:
                print(i, f"通过！num_tokens={cache_result['num_tokens']}", flush=True)
                f_w.write(line)
            else:
                wrong_cnt += 1
                print(i, f"不通过！num_tokens={cache_result['num_tokens']}", flush=True)
                
        print(f"wrong_cnt={wrong_cnt}")