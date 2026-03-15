import os
from unittest import TestCase
import torch
from xtuner.v1.datasets import VideoChat3TokenizeFnConfig
from xtuner.v1.datasets.mllm_tokenize_fn.base_mllm_tokenize_fn import collect_image_video_paths_and_extra
from transformers import AutoTokenizer, AutoProcessor
import json
import parametrize

LOCAL_MEDIA_ROOT = "tests/resource"
CEPH_ROOT = "hhdd:s3://datapipe/frame/bee_15m/Caption/COYO-Recaption/"
# VIDEOCHAT3_PATH = os.environ.get("VIDEOCHAT3_PATH", "VideoChat3-2B")
VIDEOCHAT3_PATH = "/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/VideoChat3-4B"

dd_vision_id = False

self_tokenizer = AutoTokenizer.from_pretrained(VIDEOCHAT3_PATH, trust_remote_code=True)
# self.tokenize_fn = VideoChat3TokenizeFnConfig(processor_path=VIDEOCHAT3_PATH).build(self.tokenizer)
# self.processor = AutoProcessor.from_pretrained(VIDEOCHAT3_PATH, trust_remote_code=True)
sample_max_length = 8192 * 2
tokenize_fn = tokenize_fn = VideoChat3TokenizeFnConfig(
                    max_length=sample_max_length,
                    image_min_pixels=28*28,
                    image_max_pixels=int(sample_max_length * 0.7 * 28 * 28),
                    frame_min_pixels=28*28,
                    frame_max_pixels=int(640*480),
                    video_max_total_pixels=int(sample_max_length * 0.6 * 4 * 28 * 28),
                    video_min_frames=1,
                    video_max_frames=256, 
                    fixed_num_sampled_frames=None,
                    video_sample_fps=2, 
                    processor_path=VIDEOCHAT3_PATH,
                    data_augment=False,
                    system_message=None,
                    hash=None,
                    ).build(self_tokenizer)
data_path = '/mnt/petrelfs/zengxiangyu/Research_lixinhao/videochat3_data_annotations/image/beedata_annos_newest_no_think_caption_part/Caption_COYO-Recaption_train_1091735.jsonl'
output_path = data_path.replace('.jsonl', '_cleaned2.jsonl')

success_count = 0
fail_count = 0
failed_indices = []

with open(data_path, encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
    for i, line in enumerate(f_in):
        raw_data_copy = json.loads(line)
        tokenize_fn.state = "cache"
        try:
            cache_result = tokenize_fn(raw_data_copy, media_root=LOCAL_MEDIA_ROOT)
            # 成功的行，写入新文件
            f_out.write(line)
            success_count += 1
            if success_count % 1000 == 0:
                print(f"已处理 {i+1} 行，成功 {success_count} 行，失败 {fail_count} 行", flush=True)
        except Exception as e:
            fail_count += 1
            failed_indices.append(i)
            print(i, f"失败！error={e}, id={raw_data_copy.get('id', 'unknown')}", flush=True)
            continue

print(f"\n处理完成！")
print(f"总行数: {success_count + fail_count}")
print(f"成功: {success_count} 行")
print(f"失败: {fail_count} 行")
print(f"输出文件: {output_path}")
if failed_indices:
    print(f"失败的行号: {failed_indices[:20]}{'...' if len(failed_indices) > 20 else ''}")
    