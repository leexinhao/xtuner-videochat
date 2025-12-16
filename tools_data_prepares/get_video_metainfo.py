import random
import os
import io
# import av
import cv2
import decord
import imageio
from decord import VideoReader
import numpy as np
import math

from torchvision.transforms.functional import pil_to_tensor
from petrel_client.client import Client
import json
import yaml
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed



def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets



def pts_to_secs(pts: int, time_base: float, start_pts: int) -> float:
    """
    Converts a present time with the given time base and start_pts offset to seconds.

    Returns:
        time_in_seconds (float): The corresponding time in seconds.

    https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/data/utils.py#L54-L64
    """
    if pts == math.inf:
        return math.inf

    return int(pts - start_pts) * time_base




def read_frames_av(video_path, client=None, clip=None):
    if clip is not None:
        raise NotImplementedError("av don't support clip!!!")
    if 's3://' in video_path:
        video_bytes = client.get(video_path)
        byteio = io.BytesIO(video_bytes)
        byteio.seek(0)
        reader = av.open(byteio)
    else:
        byteio = None
        reader = av.open(video_path)

    video_stream = video_reader.streams.video[0]
    video_duration = float(pts_to_secs(
        video_stream.duration,
        video_stream.time_base,
        video_stream.start_time
    ))
    
    
    if video_stream.frames is not None and video_stream.frames > 0:
        total_num_frames= video_stream.frames
    else:
        # 若 frames 无效，通过 时长×帧率 估算（可能有误差）
        meta['total_num_frames'] = int(round(video_duration * float(video_stream.average_rate)))
    
    return {
        "total_num_frames": total_num_frames,
        "fps": float(video_stream.average_rate) ,
        "width": video_stream.width,
        "height": video_stream.height,
        "duration": video_duration,
        "video_backend": "av",
        "clip_start_time": clip[0] if clip is not None else None,
        "clip_end_time": clip[1] if clip is not None else None,
    }


def read_frames_gif(
        video_path, client=None, clip=None
    ):
    raise NotImplementedError
    if clip is not None:
        raise NotImplementedError("Gif don't support clip!!!")
    if 's3://' in video_path:
        video_bytes = client.get(video_path)
        byteio = io.BytesIO(video_bytes)
        gif = imageio.get_reader(byteio)
    else:
        byteio = None
        gif = imageio.get_reader(video_path)
    vlen = len(gif)
    fps = 1.
    duration = vlen / fps
    return duration # for tgif



def read_frames_decord(
        video_path, client=None, clip=None
    ):

    if video_path.endswith('.avi'):
        return read_frames_av(video_path=video_path, 
                    client=client, clip=clip)
    if 's3://' in video_path:
        video_bytes = client.get(video_path)
        if video_bytes is None or len(video_bytes) == 0:
            raise ValueError(f"Can't read byte from {video_path}!")
        byteio = io.BytesIO(video_bytes)
        video_reader = VideoReader(byteio, num_threads=1)
    else:
        byteio = None
        video_reader = VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)

    frame_shape = video_reader[0].shape

    # 帧的宽度和高度是尺寸的后两个维度
    height = frame_shape[0]
    width = frame_shape[1]

    return {
        "total_num_frames": vlen,
        "fps": float(fps),
        "width": width,
        "height": height,
        "duration": duration,
        "video_backend": "decord",
        "clip_start_time": clip[0] if clip is not None else None,
        "clip_end_time": clip[1] if clip is not None else None,
    }


def read_frames_img(
        video_path, client=None, clip=None
    ):
    img_list = []
    if "s3://" in video_path:
        all_paths = client.list(video_path)
        for path in all_paths:
            img_list.append(path)
        # 读取第一张图片获取宽高
        width, height = None, None
        if len(img_list) > 0:
            first_img_bytes = client.get(img_list[0])
            with Image.open(io.BytesIO(first_img_bytes)) as img:
                width, height = img.size
    else:
        all_paths = [os.path.join(video_path, p) for p in os.listdir(video_path)]
        for path in all_paths:
            img_list.append(path)
        # 读取第一张图片获取宽高
        width, height = None, None
        if len(img_list) > 0:
            with Image.open(img_list[0]) as img:
                width, height = img.size


    if clip is not None:
        vlen = float(clip[1]) - float(clip[0])
    else:
        vlen = len(img_list)

    return {
        "total_num_frames": vlen,
        "fps": None,
        "width": width,
        "height": height,
        "duration": None,
        "video_backend": "img",
        "clip_start_time": clip[0] if clip is not None else None,
        "clip_end_time": clip[1] if clip is not None else None,
    }


VIDEO_READER_FUNCS = {
    # 'av': read_frames_av,
    'decord': read_frames_decord,
    # 'gif': read_frames_gif,
    'img': read_frames_img,
    'frame': read_frames_img,
    # 'lazy': read_frames_decord
}



def get_video_meta_info(video_file, data_anno, client, video_reader_type):
    # print(f"\n\nInspecting the video path, video_file={video_file}\n\n")
    # start_time = time.time()

    if "start" in data_anno and "end" in data_anno:
        clip = [data_anno["start"], data_anno["end"]]
    else:
        clip = None

    if clip is None or video_reader_type == "img":
        video_reader = VIDEO_READER_FUNCS[video_reader_type]
        video_meta_info = video_reader(
            video_file, client=client, clip=clip,
        )

    else:
        raise NotImplementedError("目前不支持对原始video进行裁剪")
        # video_reader = VIDEO_READER_FUNCS['lazy']
        # start, end = clip
        # duration = end - start

    video_meta_info['video_path'] = video_file
    video_meta_info['video_reader_type'] = video_reader_type

    return video_meta_info

client = Client(conf_path='~/petreloss.conf')
data_path = "/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/tools_data_prepares/data_list_example_llava_video.json"
save_root = "/mnt/petrelfs/zengxiangyu/Research_lixinhao/vflash_annotations/video_metainfos_new"

os.makedirs(save_root, exist_ok=True)

with open(data_path, "r") as froot:
    datasets = json.load(froot)
    for dataset_name in datasets:
        _data = datasets[dataset_name]
        anno_path =_data['anno_path']
        save_path = os.path.join(save_root, dataset_name + '_metainfos.jsonl')
        # os.makedirs(save_dir, exist_ok=True)

        print(f"Loading {anno_path}, save to {save_path}")

        if anno_path.endswith(".jsonl"):
            cur_data_dict = []
            if "s3://" in anno_path:
                with io.BytesIO(client.get(anno_path)) as json_file:
                    for line in json_file:
                        cur_data_dict.append(json.loads(line.strip()))
            else:
                with open(anno_path, "r") as json_file:
                    for line in json_file:
                        cur_data_dict.append(json.loads(line.strip()))
        elif anno_path.endswith(".json"):
            if "s3://" in anno_path:
                raise NotImplementedError(anno_path)
                with io.BytesIO(client.get(anno_path)) as json_file:
                    cur_data_dict = json.load(json_file)
            else:
                with open(anno_path, "r") as json_file:
                    cur_data_dict = json.load(json_file)
        else:
            raise ValueError(f"Unsupported file type: {anno_path}")
        
        video_read_type = _data.get("video_read_type", 'decord')
        data_root = _data.get("media_root", '')

        media_type = 'video'
        if 'video' not in cur_data_dict[0].keys():
            continue
        new_cur_data_dict = []
        error_logs = []
        num_workers = int(min(16, os.cpu_count() or 16))

        def _process_one(record):
            old_path = record[media_type]
            new_path = os.path.join(data_root, old_path)
            meta = get_video_meta_info(
                new_path,
                data_anno=record,
                client=client,
                video_reader_type=video_read_type,
            )
            return meta

        # 读取已存在的结果，避免重复处理；并确定写入模式（续写/新写）
        processed_paths = set()
        write_mode = "w"
        if os.path.exists(save_path):
            print(f"跳过: {save_path}")
            continue
            try:
                with open(save_path, "r") as existed_f:
                    for line in existed_f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                            vp = row.get("video_path")
                            if vp:
                                processed_paths.add(vp)
                        except Exception:
                            # 跳过坏行
                            continue
                write_mode = "a"
            except Exception:
                # 无法读取则从头重写
                write_mode = "w"

        # 过滤已处理的数据，并按 target_path 去重
        to_process_map = {}
        for d in cur_data_dict:
            target_path = os.path.join(data_root, d[media_type])
            if target_path in processed_paths:
                continue
            # 避免同一 target_path 在标注中重复
            if target_path not in to_process_map:
                to_process_map[target_path] = d
        to_process = list(to_process_map.values())

        # 无需处理则跳过执行器，仅保证错误文件可能存在的追加
        if len(to_process) == 0:
            continue

        # 实时写入结果到 jsonl（支持断点续写）
        with open(save_path, write_mode) as out_f:
            err_f = None
            try:
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    future_to_item = {executor.submit(_process_one, d): d for d in to_process}
                    for future in tqdm(as_completed(future_to_item), total=len(to_process)):
                        item = future_to_item[future]
                        old_data_path = item[media_type]
                        try:
                            result = future.result()
                            # 立即写入一行
                            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                            out_f.flush()
                            # 可选：保留内存列表
                            new_cur_data_dict.append(result)
                        except Exception as e:
                            err_msg = f"{old_data_path} | {repr(e)}"
                            error_logs.append({"video": old_data_path, "error": repr(e)})
                            # 立即写入错误日志
                            if err_f is None:
                                err_path = os.path.splitext(save_path)[0] + "_errors.jsonl"
                                err_f_mode = "a" if os.path.exists(err_path) else "w"
                                err_f = open(err_path, err_f_mode)
                            err_f.write(json.dumps({"video": old_data_path, "error": repr(e)}, ensure_ascii=False) + "\n")
                            err_f.flush()
                            try:
                                tqdm.write(err_msg)
                            except Exception:
                                print(err_msg)
            finally:
                if err_f is not None:
                    err_f.close()