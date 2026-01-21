import os
from unittest import TestCase
import torch
from xtuner.v1.datasets import VideoChat3TokenizeFnConfig
from xtuner.v1.datasets.mllm_tokenize_fn.base_mllm_tokenize_fn import collect_image_video_paths_and_extra
from transformers import AutoTokenizer, AutoProcessor
import json
import parametrize

LOCAL_MEDIA_ROOT = "tests/resource"
CEPH_ROOT = "pnorm2:s3://videochat3_frames/llava-video-178k_frames_fps4/"
# VIDEOCHAT3_PATH = os.environ.get("VIDEOCHAT3_PATH", "VideoChat3-2B")
VIDEOCHAT3_PATH = "/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/work_dir/stage1-2/VideoChat3_4B_train_stage1-2_minix2_32k_lr8e-5/20260119023400/hf-581"
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
data_path = '/mnt/petrelfs/zengxiangyu/Research_lixinhao/videochat3_data_annotations/llava_video_xtuner_format_20251216_youtube/llava-video_30_60_s_youtube_mc_v0_1_qa_processed_39927_qwen3vl235b_rewrtten.jsonl'
with open(data_path, encoding='utf-8') as f:
    for i, line in enumerate(f):
        # if i >= 10:
        #     break
        # raw_data = json.loads(line)
        # tokenize_fn.state = "get_item"
        # # ret_xtuner = tokenize_fn(raw_data, media_root=LOCAL_MEDIA_ROOT)
        # ret_xtuner = tokenize_fn(raw_data, media_root=CEPH_ROOT)
        # input_ids_xtuner = ret_xtuner['input_ids']
        # pixel_values_xtuner: torch.Tensor = ret_xtuner['pixel_values']
        # video_grid_thw_xtuner: torch.Tensor = ret_xtuner['image_grid_thw']
        
        raw_data_copy = json.loads(line)
        tokenize_fn.state = "cache"
        try:
            cache_result = tokenize_fn(raw_data_copy, media_root=LOCAL_MEDIA_ROOT)
        except Exception as e:
            print(i, f"失败！error={e}, {raw_data_copy}", flush=True)
            continue

        # assert len(input_ids_xtuner) == cache_result['num_tokens'], f"calc_num_tokens_get_item{cache_result['num_tokens']}和get_item出来的token数{len(input_ids_xtuner)}不一致！"
        # print(i, f"通过！num_tokens={cache_result['num_tokens']}", flush=True)
    