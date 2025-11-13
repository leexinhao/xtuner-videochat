import random
import os
import io
# import av
import cv2
import imageio
import numpy as np
import math

from torchvision.transforms.functional import pil_to_tensor
from petrel_client.client import Client
import json
import yaml
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed



def read_img(image_path, client=None):

    if "s3://" in image_path:
        img_bytes = client.get(image_path)
        with Image.open(io.BytesIO(img_bytes)) as img:
            width, height = img.size
    else:
        with Image.open(image_path) as img:
            width, height = img.size


    return {
        "width": width,
        "height": height,
    }



def get_image_meta_info(image_file, data_anno, client):

    image_meta_info = read_img(image_file, client=client)


    image_meta_info['image_path'] = image_file

    return image_meta_info

client = Client(conf_path='~/petreloss.conf')
data_path = "/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/tools_data_prepares/data_llava585k.json"
save_root = "/mnt/hwfile/zengxiangyu/Research_lixinhao/vflash_annotations/image_metainfos"

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
        

        data_root = _data.get("media_root", '')

        media_type = 'image'
        if 'image' not in cur_data_dict[0].keys():
            continue
        new_cur_data_dict = []
        error_logs = []
        num_workers = int(min(16, os.cpu_count() or 16))

        def _process_one(record):
            old_path = record[media_type]
            new_path = os.path.join(data_root, old_path)
            meta = get_image_meta_info(
                new_path,
                data_anno=record,
                client=client
            )
            return meta

        # 读取已存在的结果，避免重复处理；并确定写入模式（续写/新写）
        processed_paths = set()
        write_mode = "w"
        if os.path.exists(save_path):
            try:
                with open(save_path, "r") as existed_f:
                    for line in existed_f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                            vp = row.get("image_path")
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
                            error_logs.append({"image": old_data_path, "error": repr(e)})
                            # 立即写入错误日志
                            if err_f is None:
                                err_path = os.path.splitext(save_path)[0] + "_errors.jsonl"
                                err_f_mode = "a" if os.path.exists(err_path) else "w"
                                err_f = open(err_path, err_f_mode)
                            err_f.write(json.dumps({"image": old_data_path, "error": repr(e)}, ensure_ascii=False) + "\n")
                            err_f.flush()
                            try:
                                tqdm.write(err_msg)
                            except Exception:
                                print(err_msg)
            finally:
                if err_f is not None:
                    err_f.close()