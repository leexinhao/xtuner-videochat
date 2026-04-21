#!/usr/bin/env python3
"""统计 data json（如 data_stage2_image_video_minisft_v22.json）中所有 anno_path 指向的 jsonl 行数。"""

import argparse
import json
import math
import os
from typing import Any, List, Optional


def _normalize_anno_paths(anno_path: Any) -> List[str]:
    if isinstance(anno_path, str):
        return [anno_path] if anno_path else []
    if isinstance(anno_path, list):
        return [p for p in anno_path if isinstance(p, str) and p]
    return []


def count_lines_fast(path: str) -> int:
    """按换行符计数（与 wc -l 一致：文件末尾无换行时比「json 对象数」少 1 的可能存在）。"""
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            n += chunk.count(b"\n")
    return n


def _sample_ratio(cfg: dict) -> float:
    r = cfg.get("sample_ratio", 1.0)
    if isinstance(r, (int, float)):
        return float(r)
    try:
        return float(r)
    except (TypeError, ValueError):
        return 1.0


def _used_line_count(line_count: int, sample_ratio: float) -> Optional[int]:
    """按训练配置的 sample_ratio 估算实际会用到的行数（向下取整）。"""
    if line_count < 0:
        return None
    return int(math.floor(line_count * sample_ratio))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_json",
        nargs="?",
        default="/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/training_data_annotations/data_stage2_image_video_minisft_v22.json",
        help="含各数据集 anno_path 的配置 json",
    )
    parser.add_argument(
        "--no-per-dataset",
        action="store_true",
        help="不打印每个数据集一行，只打印汇总",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="将统计结果保存到 JSON 文件，例如 /tmp/count_result.json",
    )
    args = parser.parse_args()

    with open(args.data_json, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise SystemExit("根节点应为 dict：{ 数据集名: { anno_path: ... } }")

    path_to_lines: dict[str, int] = {}
    path_to_keys: dict[str, list[str]] = {}
    missing: list[tuple[str, str]] = []

    for key, cfg in data.items():
        if not isinstance(cfg, dict):
            continue
        paths = _normalize_anno_paths(cfg.get("anno_path"))
        if not paths:
            continue
        for p in paths:
            path_to_keys.setdefault(p, []).append(key)
            if p in path_to_lines:
                continue
            if not os.path.isfile(p):
                path_to_lines[p] = -1
                missing.append((key, p))
                continue
            path_to_lines[p] = count_lines_fast(p)

    per_dataset_rows = []
    if not args.no_per_dataset:
        for key, cfg in sorted(data.items(), key=lambda x: x[0]):
            if not isinstance(cfg, dict):
                continue
            paths = _normalize_anno_paths(cfg.get("anno_path"))
            if not paths:
                continue
            ratio = _sample_ratio(cfg)
            for p in paths:
                n = path_to_lines.get(p, -1)
                used = _used_line_count(n, ratio)
                status = str(n) if n >= 0 else "MISSING"
                used_s = str(used) if used is not None else "NA"
                print(f"{status}\t{used_s}\t{ratio}\t{key}\t{p}")
                per_dataset_rows.append(
                    {
                        "dataset_key": key,
                        "anno_path": p,
                        "sample_ratio": ratio,
                        "line_count": n,
                        "used_line_count": used,
                        "status": "ok" if n >= 0 else "missing",
                    }
                )
    else:
        for key, cfg in sorted(data.items(), key=lambda x: x[0]):
            if not isinstance(cfg, dict):
                continue
            paths = _normalize_anno_paths(cfg.get("anno_path"))
            if not paths:
                continue
            ratio = _sample_ratio(cfg)
            for p in paths:
                n = path_to_lines.get(p, -1)
                used = _used_line_count(n, ratio)
                per_dataset_rows.append(
                    {
                        "dataset_key": key,
                        "anno_path": p,
                        "sample_ratio": ratio,
                        "line_count": n,
                        "used_line_count": used,
                        "status": "ok" if n >= 0 else "missing",
                    }
                )

    total_unique = sum(n for n in path_to_lines.values() if n >= 0)
    n_files = sum(1 for n in path_to_lines.values() if n >= 0)
    n_missing = sum(1 for n in path_to_lines.values() if n < 0)
    total_used = sum(
        row["used_line_count"]
        for row in per_dataset_rows
        if row.get("used_line_count") is not None
    )

    print()
    print(f"data_json: {args.data_json}")
    print(f"数据集条目数（含 anno_path）: {sum(1 for k, c in data.items() if isinstance(c, dict) and _normalize_anno_paths(c.get('anno_path')))}")
    print(f"唯一 jsonl 路径数: {len(path_to_lines)}")
    print(f"存在且已计行数的文件: {n_files}")
    print(f"缺失或非文件: {n_missing}")
    print(f"所有唯一 jsonl 行数之和: {total_unique}")
    print(
        "按 sample_ratio 估算的 used_line_count 之和（每条数据集一行分别乘 ratio 后向下取整再相加）: "
        f"{total_used}"
    )

    dupes = {p: ks for p, ks in path_to_keys.items() if len(ks) > 1}
    if dupes:
        print(f"注意: 有 {len(dupes)} 个 anno_path 被多个数据集名引用（上表按路径去重后只计一次行数）。")

    if args.output_json:
        result = {
            "data_json": args.data_json,
            "summary": {
                "dataset_entries_with_anno_path": sum(
                    1 for _, c in data.items()
                    if isinstance(c, dict) and _normalize_anno_paths(c.get("anno_path"))
                ),
                "unique_jsonl_paths": len(path_to_lines),
                "existing_files": n_files,
                "missing_files": n_missing,
                "total_unique_lines": total_unique,
                "total_used_line_count": total_used,
                "used_line_count_note": (
                    "各数据集 line_count * sample_ratio 向下取整后求和；"
                    "若多个数据集共用同一 jsonl，该和可能大于按文件去重后的加权行数。"
                ),
                "duplicate_anno_path_count": len(dupes),
            },
            "per_dataset": per_dataset_rows,
            "duplicate_anno_paths": dupes,
            "missing_paths": [{"dataset_key": k, "anno_path": p} for k, p in missing],
        }
        output_dir = os.path.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"JSON 已保存: {args.output_json}")


if __name__ == "__main__":
    main()
