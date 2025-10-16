import os
from unittest import TestCase
import torch
from xtuner.v1.datasets import VideoChat3TokenizeFnConfig
from transformers import AutoTokenizer, AutoProcessor
import json
import parametrize

VIDEOCHAT3_PATH = os.environ.get("VIDEOCHAT3_PATH", "VideoChat3-2B")



class TestVideoChat3TokenizeFn(TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained(VIDEOCHAT3_PATH, trust_remote_code=True)
        self.tokenize_fn = VideoChat3TokenizeFnConfig(processor_path=VIDEOCHAT3_PATH).build(self.tokenizer)
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
                messages[0]['content'][0]['type'] = 'image'
                messages[0]['content'][0]['path'] = 'tests/' + messages[0]['content'][0]['image_url']['url']
                del messages[0]['content'][0]['image_url']

                messages[0]['content'][1]['text'] = messages[0]['content'][1]['text'].replace('<IMG_CONTEXT>', '')
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
                
                # 检查calc_num_tokens_get_item和get_item出来的token数是否一致
                # 需要加载原始数据备份，因为raw_data被修改了
                raw_data_copy = json.loads(line)
                self.tokenize_fn.state = "cache"
                cache_result = self.tokenize_fn(raw_data_copy, media_root='tests/')
                self.tokenize_fn.state = "get_item"
                self.assertEqual(len(input_ids_xtuner), cache_result['num_tokens'], 
                                "calc_num_tokens_get_item和get_item出来的token数不一致")

    @parametrize.parametrize("add_vision_id", [(True,), (False,)])
    def test_videochat3_multi_image(self, add_vision_id):
        tokenize_fn = VideoChat3TokenizeFnConfig(processor_path=VIDEOCHAT3_PATH,
                                              add_vision_id=add_vision_id).build(self.tokenizer)
        data_path = 'tests/resource/mllm_sft_multi_image_example_data.jsonl'
        total_step = 5
        with open(data_path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= total_step:
                    break
                raw_data = json.loads(line)
                
                # \n 必须去掉，否则和 hf 无法对齐
                messages = raw_data['messages']
                messages[0]['content'][2]['text'] = messages[0]['content'][2]['text'].replace('\n', '')

                ret = tokenize_fn(raw_data, media_root='tests/')
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
                messages[0]['content'][2]['text'] = messages[0]['content'][2]['text'].replace('<IMG_CONTEXT>', '')
                for msg in messages:
                    if not isinstance(msg['content'], list):
                        msg['content'] = [{"type": "text", "text": msg['content']}]

                ret = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=True,
                                                         return_dict=True, add_vision_id=add_vision_id)
                input_ids_hf = ret['input_ids'][0]
                pixel_values_hf = ret['pixel_values']
                image_grid_thw_hf = ret['image_grid_thw']
                self.assertEqual(input_ids_xtuner, input_ids_hf)
                self.assertTrue(torch.allclose(pixel_values_xtuner, pixel_values_hf))
                self.assertTrue(torch.allclose(image_grid_thw_xtuner, image_grid_thw_hf))
                
                # 检查calc_num_tokens_get_item和get_item出来的token数是否一致
                # 需要加载原始数据备份，因为raw_data被修改了
                # cache状态需要保持<IMG_CONTEXT>占位符，因为calc_num_tokens_image_get_item需要它们
                raw_data_for_cache = json.loads(line)
                messages_for_cache = raw_data_for_cache['messages']
                # 只删除\n，但保持<IMG_CONTEXT>占位符
                messages_for_cache[0]['content'][2]['text'] = messages_for_cache[0]['content'][2]['text'].replace('\n', '')
                
                tokenize_fn.state = "cache"
                cache_result = tokenize_fn(raw_data_for_cache, media_root='tests/')
                tokenize_fn.state = "get_item"
                self.assertEqual(len(input_ids_xtuner), cache_result['num_tokens'], 
                               "calc_num_tokens_get_item和get_item出来的token数不一致")

    @parametrize.parametrize("add_vision_id", [(True,), (False,)])
    def test_videochat3_video(self, add_vision_id):
        tokenize_fn = VideoChat3TokenizeFnConfig(processor_path=VIDEOCHAT3_PATH,
                                              add_vision_id=add_vision_id).build(self.tokenizer)
        data_path = 'tests/resource/mllm_sft_video_example_data.jsonl'
        with open(data_path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 1:
                    break
                raw_data = json.loads(line)

                # 测试collect_image_video_paths_and_extra函数
                from xtuner.v1.datasets.mllm_tokenize_fn.base_mllm_tokenize_fn import collect_image_video_paths_and_extra
                image_paths, video_paths, extra_info = collect_image_video_paths_and_extra(raw_data['messages'])
                
                # 验证路径和元数据提取正确
                self.assertEqual(len(video_paths), 1)
                self.assertEqual(video_paths[0], 'resource/tennis.mp4')
                self.assertEqual(len(extra_info['video_meta_list']), 1)
                self.assertIsNotNone(extra_info['video_meta_list'][0])
                
                # 验证消息结构
                self.assertIn('messages', raw_data)
                self.assertIsInstance(raw_data['messages'], list)
                self.assertGreater(len(raw_data['messages']), 0)
                # 对xtuner的结果
                ret_xtuner = tokenize_fn(raw_data, media_root='tests/')
                input_ids_xtuner = ret_xtuner['input_ids']
                pixel_values_xtuner: torch.Tensor = ret_xtuner['pixel_values']
                video_grid_thw_xtuner: torch.Tensor = ret_xtuner['image_grid_thw']

                # 转为hf openai格式
                messages = raw_data['messages']
                for msg in messages:
                    if not isinstance(msg['content'], list):
                        msg['content'] = [{"type": "text", "text": msg['content']}]
                for msg in messages:
                    for c in msg['content']:
                        if c.get('type') == 'video_url':
                            # 转为HF所需的video type
                            c['type'] = 'video'
                            c['path'] = 'tests/' + c['video_url']['url']
                            del c['video_url']
                        elif c.get('type') == 'text' and '<VIDEO_CONTEXT>' in c['text']:
                            # 移除<VIDEO_CONTEXT>占位符，因为HF会直接处理video token
                            c['text'] = c['text'].replace('<VIDEO_CONTEXT>', '')

                if len(video_paths) == 1:
                    ret_hf = self.processor.apply_chat_template(
                        messages, add_generation_prompt=False, tokenize=True,
                        return_dict=True, add_vision_id=False)
                else:
                    ret_hf = self.processor.apply_chat_template(
                        messages, add_generation_prompt=False, tokenize=True,
                        return_dict=True, add_vision_id=add_vision_id)
                input_ids_hf = ret_hf['input_ids'][0]
                pixel_values_hf = ret_hf['pixel_values_videos']
                video_grid_thw_hf = ret_hf['video_grid_thw']

                self.assertEqual(input_ids_xtuner, input_ids_hf)
                self.assertTrue(torch.allclose(pixel_values_xtuner, pixel_values_hf))
                self.assertTrue(torch.allclose(video_grid_thw_xtuner, video_grid_thw_hf))
                
                # 检查calc_num_tokens_get_item和get_item出来的token数是否一致
                # 需要重新加载原始数据，因为raw_data被修改了
                # cache状态需要保持<VIDEO_CONTEXT>占位符，因为calc_num_tokens_video_get_item需要它们
                raw_data_copy = json.loads(line)
                tokenize_fn.state = "cache"
                cache_result = tokenize_fn(raw_data_copy, media_root='tests/')
                tokenize_fn.state = "get_item"
                self.assertEqual(len(input_ids_xtuner), cache_result['num_tokens'], 
                                "calc_num_tokens_get_item和get_item出来的token数不一致")


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
                for msg in messages:
                    if not isinstance(msg['content'], list):
                        msg['content'] = [{"type": "text", "text": msg['content']}]

                ret = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=True,
                                                         return_dict=True)
                input_ids_hf = ret['input_ids'][0]
                assert input_ids_xtuner == input_ids_hf
                
                # 检查calc_num_tokens_get_item和get_item出来的token数是否一致
                # 需要重新加载原始数据，因为raw_data被修改了
                original_data = None
                with open(data_path, encoding='utf-8') as f:
                    for j, line in enumerate(f):
                        if j == i:
                            original_data = json.loads(line)
                            break
                if original_data is not None:
                    self.tokenize_fn.state = "cache"
                    cache_result = self.tokenize_fn(original_data)
                    self.tokenize_fn.state = "get_item"
                    self.assertEqual(len(input_ids_xtuner), cache_result['num_tokens'], 
                                   "calc_num_tokens_get_item和get_item出来的token数不一致")
