import json
import os

json_meta_path = "/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/training_data_annotations/debug/data_stage2_image_video_minisft_v4_debug.json"
with open(json_meta_path, 'r') as f:
    metadata = json.load(f)



for json_name in metadata.keys():
    print("json_name:", json_name)
    wrong_cnt = 0
    jsonl_path = metadata[json_name]['anno_path']
    save_path = jsonl_path.replace('.jsonl', '_clean.jsonl')
    with open(jsonl_path, 'r') as f:
        with open(save_path, 'w') as f_w:
            for i, line in enumerate(f):
                _data = json.loads(line)
                data = _data['messages'][0]['content'][0]['video_metadata']
                
                # print(data)
                try:
                    assert data['total_num_frames'] > 0, "必须有视频帧"
                    if data['fps'] is not None and data['duration'] is not None:
                        if data['duration'] < 1:
                             raise ValueError(f"过滤掉低于1s的视频 {data['duration']}")
                        expected_frames = data['fps'] * data['duration']
                        if abs(expected_frames - data['total_num_frames']) > 1e-6:
                            raise ValueError(f"fps * duration must be equal to total_num_frames, but got {expected_frames} != {data['total_num_frames']}")
                    
                    if 'video_start_time' in data.keys():
                        if data['video_start_time'] < 0 or (data['duration'] is not None and data['video_start_time'] >= data['duration']):
                            raise ValueError(f"video_start_time must be greater than or equal to 0 and less than duration, but got {data['video_start_time']}")
                    if 'clip_start_time' in data.keys():
                        if (data['clip_start_time'] is None) != (data['clip_end_time'] is None):
                            raise ValueError("clip_start_time and clip_end_time must both be None or both be not None.")
                        if data['clip_start_time'] is not None and data['clip_end_time'] is not None:
                            if data['clip_end_time'] <= data['clip_start_time']:
                                raise ValueError(f"clip_end_time must be greater than clip_start_time, but got {data['clip_end_time']} <= {data['clip_start_time']}")
                            
                            if data['clip_end_time'] <= (data['clip_start_time'] + 1):
                                raise ValueError(f"过滤掉低于1s的视频， but got {data['clip_end_time']} <= 1 + {data['clip_start_time']}")

                            if data['clip_start_time'] < 0:
                                raise ValueError(f"clip_start_time must be greater than or equal to 0, but got {data['clip_start_time']}")
                            
                            # 修复clip_end_time超出范围的情况
                            if data['clip_end_time'] < 0:
                                raise ValueError(f"clip_end_time must be greater than or equal to 0, but got {data['clip_end_time']}")
                            
                            if data['duration'] is not None and data['clip_end_time'] > data['duration']:
                                raise ValueError(f"clip_end_time must be less than or equal to duration ({data['duration']}), but got {data['clip_end_time']}")
                            
                            # 最终验证：确保修复后的clip_end_time仍然满足所有条件
                            if data['clip_end_time'] <= (data['clip_start_time'] + 1):
                                raise ValueError(f"修复后clip_end_time仍然无效: {data['clip_end_time']} <= 1 + {data['clip_start_time']}")
                            
                            if data['duration'] is not None and data['clip_end_time'] > data['duration']:
                                raise ValueError(f"修复后clip_end_time仍然无效: {data['clip_end_time']} > {data['duration']}")
                            
                except Exception as e:
                    print(f"{json_name}第{i}条数据不通过！")
                    wrong_cnt += 1
                    print(e)
                    continue
                    
                f_w.write(json.dumps(_data) + '\n')
                # print(f"{json_name}第{i}条数据通过！")


    print(json_name, 'wrong_cnt:', wrong_cnt)