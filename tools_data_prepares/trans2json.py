import os
import json
import yaml

yaml_path = "/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/tools_data_prepares/data_list_example_llava_video.yaml"
json_out_path = "/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/tools_data_prepares/data_list_example_llava_video_sub_generated.json"

def main():
    # 读取 YAML（一个列表，每个元素是 dict）
    with open(yaml_path, "r", encoding="utf-8") as f:
        data_list = yaml.safe_load(f)

    result = {}

    for item in data_list:
        anno_path = item["json_path"]
        media_root = item["data_root"]
        sampling_strategy = item.get("sampling_strategy", "all")

        # 从文件名生成 key：
        # /path/.../llava-video_xxx_with_duration.jsonl
        # -> llava-video_xxx
        base = os.path.basename(anno_path)
        key = base.replace("_with_duration", "").replace(".jsonl", "")

        # 构造与 data_list_example_llava_video_sub.json 相同结构
        result[key] = {
            "media_root": media_root,
            "anno_path": anno_path,
            # length 在示例 JSON 中是真实样本数，这里先填 0 占位，
            # 如需严格一致，可再写脚本统计每个 jsonl 的行数再填回去。
            "length": 0,
            "data_augment": False,
            "repeat_time": 1,
            "sampling_strategy": sampling_strategy,
        }

    # 写出 JSON
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print(f"转换完成，已保存到: {json_out_path}")

if __name__ == "__main__":
    main()