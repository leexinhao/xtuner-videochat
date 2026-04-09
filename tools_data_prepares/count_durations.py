#!/usr/bin/env python3
"""
视频时长统计工具
统计JSONL文件中所有视频的duration
"""

import json
import argparse
from collections import defaultdict


def extract_duration(data: dict) -> float:
    """从数据中提取视频duration"""
    try:
        messages = data.get("messages", [])
        if not messages:
            return 0.0

        first_message = messages[0]
        content = first_message.get("content", [])
        if not content:
            return 0.0

        first_content = content[0]
        video_metadata = first_content.get("video_metadata", {})

        # 如果有clip_start_time和clip_end_time，使用它们的差值作为duration
        clip_start = video_metadata.get("clip_start_time")
        clip_end = video_metadata.get("clip_end_time")
        if clip_start is not None and clip_end is not None:
            return float(clip_end - clip_start)

        # 否则使用duration字段
        duration = video_metadata.get("duration", 0.0)
        return float(duration)
    except (KeyError, IndexError, TypeError, ValueError):
        return 0.0


def analyze_durations(jsonl_path: str):
    """分析JSONL文件中所有视频的duration统计信息"""
    durations = []
    stats = {
        "total_videos": 0,
        "min_duration": float('inf'),
        "max_duration": 0,
        "sum_duration": 0,
        "avg_duration": 0,
    }

    # 分布统计 (单位: 秒)
    distribution = defaultdict(int)

    print(f"正在分析文件: {jsonl_path}")
    print("=" * 60)

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
                duration = extract_duration(data)

                if duration > 0:
                    durations.append(duration)
                    stats["total_videos"] += 1
                    stats["sum_duration"] += duration
                    stats["min_duration"] = min(stats["min_duration"], duration)
                    stats["max_duration"] = max(stats["max_duration"], duration)

                    # 统计分布
                    if duration < 30:
                        distribution["0-30s"] += 1
                    elif duration < 60:
                        distribution["30-60s"] += 1
                    elif duration < 120:
                        distribution["60-120s"] += 1
                    elif duration < 300:
                        distribution["120s-5min"] += 1
                    elif duration < 600:
                        distribution["5-10min"] += 1
                    elif duration < 1200:
                        distribution[">10min"] += 1
                    elif duration < 3600:
                        distribution[">20min"] += 1
                    else:
                        distribution[">60min"] += 1

                if (i + 1) % 1000 == 0:
                    print(f"已处理 {i + 1} 行...")

            except json.JSONDecodeError:
                print(f"警告: 第 {i + 1} 行JSON解析失败")
                continue

    if stats["total_videos"] > 0:
        stats["avg_duration"] = stats["sum_duration"] / stats["total_videos"]
    else:
        stats["min_duration"] = 0

    # 打印统计结果
    print("\n" + "=" * 60)
    print("视频时长统计结果")
    print("=" * 60)
    print(f"总视频数: {stats['total_videos']}")
    print(f"最短时长: {stats['min_duration']:.2f} 秒 ({stats['min_duration']/60:.2f} 分钟)")
    print(f"最长时长: {stats['max_duration']:.2f} 秒 ({stats['max_duration']/60:.2f} 分钟)")
    print(f"平均时长: {stats['avg_duration']:.2f} 秒 ({stats['avg_duration']/60:.2f} 分钟)")
    print(f"总时长: {stats['sum_duration']:.2f} 秒 ({stats['sum_duration']/3600:.2f} 小时)")

    print("\n时长分布:")
    print("-" * 40)
    for range_name in ["0-30s", "30-60s", "60-120s", "120s-5min", "5-10min", ">10min", ">20min", ">60min"]:
        count = distribution[range_name]
        percentage = (count / stats["total_videos"] * 100) if stats["total_videos"] > 0 else 0
        print(f"  {range_name:12s}: {count:6d} ({percentage:5.2f}%)")

    return stats, distribution


def main():
    parser = argparse.ArgumentParser(description='视频时长统计工具')
    parser.add_argument('jsonl_path', help='输入JSONL文件路径')
    args = parser.parse_args()

    # 执行统计分析
    stats, distribution = analyze_durations(args.jsonl_path)


if __name__ == "__main__":
    main()
