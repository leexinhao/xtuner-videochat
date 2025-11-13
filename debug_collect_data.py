import json
# from xtuner.v1.datasets.mllm_tokenize_fn.base_mllm_tokenize_fn import collect_image_video_paths_and_extra

total_step = 500000
data_path = "/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/tests/resource/mllm_sft_multi_video_example_data.jsonl"
with open(data_path, encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= total_step:
            break
        raw_data = json.loads(line)

        for msg in raw_data['messages']:
            if 'role' not in msg:
                print(msg)
        # 测试collect_image_video_paths_and_extra函数
        
        # image_paths, video_paths, extra_info = collect_image_video_paths_and_extra(raw_data['messages'])
        