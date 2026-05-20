#!/usr/bin/env python3
"""统计 data json 中各 jsonl 的去重「逻辑视频」数量，并可汇总全局唯一视频键。

视频键默认规则：
- 若 url 形如 ``<prefix>/<整数>.<扩展名>``（流式切片），则逻辑视频为 ``<prefix>``；
- 否则整条相对路径视为一个逻辑视频。

全局唯一默认使用 ``(media_root, logical_video_id)``，避免不同数据集根路径下同相对路径被误合并。

性能：默认按字节正则抽取 ``video_url.url``（避免对大行 ``json.loads``）；可选 ``--jobs`` 并行扫描多个 jsonl。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Iterator, List, Set, Tuple

_CLIP_RE = re.compile(
    r"^(?P<prefix>.+)/(?P<idx>\d+)\.(?P<ext>mp4|webm|mkv|avi|mov|m4v)$",
    re.IGNORECASE,
)
_VIDEO_URL_BYTES_RE = re.compile(
    br'"video_url"\s*:\s*\{\s*"url"\s*:\s*"([^"]*)"',
)


def _normalize_anno_paths(anno_path: Any) -> List[str]:
    if isinstance(anno_path, str):
        return [anno_path] if anno_path else []
    if isinstance(anno_path, list):
        return [p for p in anno_path if isinstance(p, str) and p]
    return []


def logical_video_id_from_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    m = _CLIP_RE.match(url.replace("\\", "/"))
    if m:
        return m.group("prefix")
    return url


def _iter_video_urls(obj: Any) -> Iterator[str]:
    if isinstance(obj, dict):
        vu = obj.get("video_url")
        if isinstance(vu, dict):
            u = vu.get("url")
            if isinstance(u, str) and u:
                yield u
        for v in obj.values():
            yield from _iter_video_urls(v)
    elif isinstance(obj, list):
        for x in obj:
            yield from _iter_video_urls(x)


def unique_logical_ids_in_jsonl(path: str) -> Set[str]:
    ids: Set[str] = set()
    with open(path, "rb") as f:
        for raw in f:
            if not raw.strip():
                continue
            found = _VIDEO_URL_BYTES_RE.findall(raw)
            if found:
                for b in found:
                    u = b.decode("utf-8", errors="replace")
                    vid = logical_video_id_from_url(u)
                    if vid:
                        ids.add(vid)
                continue
            try:
                row = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            for u in _iter_video_urls(row):
                vid = logical_video_id_from_url(u)
                if vid:
                    ids.add(vid)
    return ids


def _scan_paths_parallel(paths: List[str], jobs: int) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    if jobs <= 1:
        for p in paths:
            out[p] = unique_logical_ids_in_jsonl(p)
        return out
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(unique_logical_ids_in_jsonl, p): p for p in paths}
        for fut in as_completed(futs):
            p = futs[fut]
            out[p] = fut.result()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_json",
        nargs="?",
        default="/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/training_data_annotations/stage4_online/VideoChat3_4B_train_stage4_online_v2.json",
        help="含各数据集 anno_path / media_root 的配置 json",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="将统计结果保存到 JSON 文件",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="并行读取 jsonl 的进程数；0 表示 min(32, CPU)。设为 1 关闭并行。",
    )
    args = parser.parse_args()

    with open(args.data_json, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise SystemExit("根节点应为 dict：{ 数据集名: { anno_path: ... } }")

    all_paths: Set[str] = set()
    for cfg in data.values():
        if isinstance(cfg, dict):
            all_paths.update(_normalize_anno_paths(cfg.get("anno_path")))

    existing_paths = sorted(p for p in all_paths if os.path.isfile(p))
    jobs = args.jobs
    if jobs <= 0:
        jobs = min(32, os.cpu_count() or 8)

    path_cache = _scan_paths_parallel(existing_paths, jobs)

    global_keys: Set[Tuple[str, str]] = set()
    per_dataset: List[dict[str, Any]] = []

    for key, cfg in sorted(data.items(), key=lambda x: x[0]):
        if not isinstance(cfg, dict):
            continue
        paths = _normalize_anno_paths(cfg.get("anno_path"))
        if not paths:
            continue
        media_root = cfg.get("media_root", "")
        if not isinstance(media_root, str):
            media_root = str(media_root)

        union_ids: Set[str] = set()
        missing: List[str] = []
        for p in paths:
            if not os.path.isfile(p):
                missing.append(p)
                continue
            union_ids |= path_cache.get(p, set())

        for vid in union_ids:
            global_keys.add((media_root, vid))

        per_dataset.append(
            {
                "dataset_key": key,
                "media_root": media_root,
                "anno_paths": paths,
                "missing_paths": missing,
                "unique_logical_videos": len(union_ids),
                "status": "ok" if not missing else "missing",
            }
        )

    summary = {
        "data_json": args.data_json,
        "datasets_counted": len(per_dataset),
        "unique_jsonl_paths_scanned": len(existing_paths),
        "parallel_jobs": jobs,
        "global_unique_video_keys": len(global_keys),
        "global_unique_note": (
            "global_unique_video_keys 按 (media_root, logical_video_id) 去重；"
            "若需跨数据集按相对路径合并而忽略 media_root，请自行后处理。"
        ),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    for row in per_dataset:
        print(
            f"{row['unique_logical_videos']}\t{row['status']}\t{row['dataset_key']}"
        )

    if args.output_json:
        out = {
            "summary": summary,
            "per_dataset": per_dataset,
            "global_video_keys_sample": sorted(
                [f"{mr}::{vid}" for mr, vid in global_keys]
            )[:200],
            "global_video_keys_sample_note": "仅展示前 200 条字符串形式，完整集合未写入 JSON 以免体积过大。",
        }
        od = os.path.dirname(args.output_json)
        if od:
            os.makedirs(od, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print()
        print(f"JSON 已保存: {args.output_json}")


if __name__ == "__main__":
    main()
