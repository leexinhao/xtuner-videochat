import os
from unittest import TestCase
import torch
from xtuner.v1.datasets import VideoChat3TokenizeFnConfig
from transformers import AutoTokenizer, AutoProcessor
import json

VIDEOCHAT3_PATH = os.environ.get("VIDEOCHAT3_PATH", r"C:\Users\15204\Desktop\codes\xtuner-videochat\VideoChat3-debug")
VIDEO_ROOT = os.environ.get("VIDEO_ROOT", "videos")


class TestVideoChat3TokenizeFn(TestCase):
    def setUp(self):
        tokenizer = AutoTokenizer.from_pretrained(VIDEOCHAT3_PATH, trust_remote_code=True)
        self.tokenize_fn = VideoChat3TokenizeFnConfig(processor_path=VIDEOCHAT3_PATH).build(tokenizer)
        self.processor = AutoProcessor.from_pretrained(VIDEOCHAT3_PATH, trust_remote_code=True)

    def test_videochat3_single_image(self):
        data_path = 'tests/resource/mllm_sft_single_image_example_data.jsonl'
        total_step = 5
        with open(data_path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= total_step:
                    break
                raw_data = json.loads(line)

                ret = self.tokenize_fn(raw_data, media_root='tests/')
                input_ids_xtuner = ret['input_ids']
                pixel_values_xtuner: torch.Tensor = ret['pixel_values']
                image_grid_thw_xtuner: torch.Tensor = ret['image_grid_thw']

                # to hf openai format
                messages = raw_data['messages']
                # 添加默认的system message以匹配XTuner的行为
                messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})
                messages[1]['content'][0]['type'] = 'image'
                messages[1]['content'][0]['path'] = 'tests/' + messages[1]['content'][0]['image_url']['url']
                del messages[1]['content'][0]['image_url']
                # 需要把 \n 去掉，因为 videochat3 chat_template 里面会加上 \n
                messages[1]['content'][1]['text'] = messages[1]['content'][1]['text'].replace('<IMG_CONTEXT>\n', '')
                for msg in messages:
                    if not isinstance(msg['content'], list):
                        msg['content'] = [{"type": "text", "text": msg['content']}]

                ret = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=True,
                                                         return_dict=True)
                input_ids_hf = ret['input_ids'][0]
                pixel_values_hf = ret['pixel_values']
                image_grid_thw_hf = ret['image_grid_thw']
                # 检查基本结构
                self.assertIsInstance(input_ids_xtuner, list)
                self.assertIsInstance(input_ids_hf, list)
                self.assertGreater(len(input_ids_xtuner), 0)
                self.assertGreater(len(input_ids_hf), 0)
                # 检查pixel_values和image_grid_thw的形状
                self.assertEqual(pixel_values_xtuner.shape, pixel_values_hf.shape)
                self.assertEqual(image_grid_thw_xtuner.shape, image_grid_thw_hf.shape)
                self.assertEqual(input_ids_xtuner, input_ids_hf)
                self.assertTrue(torch.allclose(pixel_values_xtuner, pixel_values_hf))
                self.assertTrue(torch.allclose(image_grid_thw_xtuner, image_grid_thw_hf))
                
    def test_videochat3_multi_image(self):
        data_path = 'tests/resource/mllm_sft_multi_image_example_data.jsonl'
        total_step = 5
        with open(data_path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= total_step:
                    break
                raw_data = json.loads(line)

                ret = self.tokenize_fn(raw_data, media_root='tests/')
                input_ids_xtuner = ret['input_ids']
                pixel_values_xtuner: torch.Tensor = ret['pixel_values']
                image_grid_thw_xtuner: torch.Tensor = ret['image_grid_thw']

                # to hf openai format
                messages = raw_data['messages']
                messages[0]['content'][0]['type'] = 'image'
                messages[0]['content'][0]['path'] = 'tests/' + messages[0]['content'][0]['image_url']['url']
                messages[0]['content'][1]['type'] = 'image'
                messages[0]['content'][1]['path'] = 'tests/' + messages[0]['content'][1]['image_url']['url']
                del messages[0]['content'][0]['image_url']
                del messages[0]['content'][1]['image_url']
                messages[0]['content'][2]['text'] = messages[0]['content'][2]['text'].replace('<IMG_CONTEXT>\n', '')
                for msg in messages:
                    if not isinstance(msg['content'], list):
                        msg['content'] = [{"type": "text", "text": msg['content']}]

                ret = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=True, return_dict=True)
                input_ids_hf = ret['input_ids'][0]
                pixel_values_hf = ret['pixel_values']
                image_grid_thw_hf = ret['image_grid_thw']
                # 检查基本结构
                self.assertIsInstance(input_ids_xtuner, list)
                self.assertIsInstance(input_ids_hf, list)
                self.assertGreater(len(input_ids_xtuner), 0)
                self.assertGreater(len(input_ids_hf), 0)
                # 检查pixel_values和image_grid_thw的形状
                self.assertEqual(pixel_values_xtuner.shape, pixel_values_hf.shape)
                self.assertEqual(image_grid_thw_xtuner.shape, image_grid_thw_hf.shape)
                self.assertEqual(input_ids_xtuner, input_ids_hf)
                self.assertTrue(torch.allclose(pixel_values_xtuner, pixel_values_hf))
                self.assertTrue(torch.allclose(image_grid_thw_xtuner, image_grid_thw_hf))

    def test_videochat3_video(self):
        data_path = 'tests/resource/mllm_sft_video_example_data.jsonl'
        with open(data_path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 1:
                    break
                raw_data = json.loads(line)

                # 由于缺少PyAV依赖，我们只测试数据解析和基本结构
                # 测试collect_image_video_paths_and_extra函数
                from xtuner.v1.datasets.mllm_tokenize_fn.base_mllm_tokenize_fn import collect_image_video_paths_and_extra
                image_paths, video_paths, extra_info = collect_image_video_paths_and_extra(raw_data['messages'])
                
                
                # 验证视频路径和元数据正确提取
                self.assertEqual(len(video_paths), 1)
                self.assertEqual(video_paths[0], 'tennis.mp4')
                self.assertEqual(len(extra_info['video_meta_list']), 1)
                self.assertIsNotNone(extra_info['video_meta_list'][0])
                
                # 验证消息结构
                self.assertIn('messages', raw_data)
                self.assertIsInstance(raw_data['messages'], list)
                self.assertGreater(len(raw_data['messages']), 0)

    def test_videochat3_pure_text(self):
        data_path = 'tests/resource/mllm_sft_text_example_data.jsonl'
        total_step = 5
        with open(data_path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= total_step:
                    break
                raw_data = json.loads(line)

                ret = self.tokenize_fn(raw_data)
                input_ids_xtuner = ret['input_ids']

                # to hf openai format
                messages = raw_data['messages']
                # 添加默认的system message以匹配XTuner的行为
                messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})
                for msg in messages:
                    if not isinstance(msg['content'], list):
                        msg['content'] = [{"type": "text", "text": msg['content']}]

                ret = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=True,
                                                         return_dict=True)
                input_ids_hf = ret['input_ids'][0]
                assert input_ids_xtuner == input_ids_hf
