#!/usr/bin/env python3
"""
每个 image_url 后面插入 {"type": "text", "text": "<IMG_CONTEXT>\n"}
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union


DEFAULT_INPUT_DIR = Path("/mnt/shared-storage-user/zengxiangyu/Research/Qwen3-VL/honey_meta_merged")
DEFAULT_OUTPUT_DIR = Path("/mnt/shared-storage-user/zengxiangyu/Research/Qwen3-VL/honey_meta_merged_with_imgcontext")

IMG_CONTEXT_TEXT = "<IMG_CONTEXT>\n"


def is_image_url_item(item: Any) -> bool:
    """检查 item 是否是 image_url 类型"""
    return isinstance(item, dict) and item.get("type") == "image_url"


def process_message_content(message: Dict[str, Any]) -> bool:
    """
    处理单个 message 的 content，在每个 image_url 后面插入 {"type": "text", "text": "<IMG_CONTEXT>\n"}
    
    返回是否进行了修改
    """
    content = message.get("content", [])
    if not isinstance(content, list):
        return False
    
    # 收集所有 image_url 的索引
    image_indices = [i for i, item in enumerate(content) if is_image_url_item(item)]
    
    if not image_indices:
        return False
    
    # 从后往前插入，避免索引变化问题
    for idx in reversed(image_indices):
        content.insert(idx + 1, {"type": "text", "text": IMG_CONTEXT_TEXT})
    
    return True


def _process_json_obj(data: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """
    对单条样本进行处理：
    遍历所有 user message，为每个 image_url 添加对应的 <IMG_CONTEXT>
    
    返回 (data, modified)
    """
    modified = False

    messages = data.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"messages is not a list: {messages}")

    for message in messages:
        if not isinstance(message, dict):
            continue
        
        # 只处理 user message
        if message.get("role") != "user":
            continue
        
        if process_message_content(message):
            modified = True

    return data, modified


def process_jsonl(input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Tuple[int, int]:
    """
    处理JSONL文件，检测 image 数量并添加 context
    
    Args:
        input_path: 输入JSONL文件路径
        output_path: 输出JSONL文件路径，如果不提供则默认添加 _with_imgcontext 后缀
    """
    input_path = Path(input_path)
    
    if output_path is None:
        # 默认输出到同名文件，加上_with_imgcontext后缀
        output_path = input_path.parent / f"{input_path.stem}_with_imgcontext{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    processed_count = 0
    total_count = 0
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Processing {input_path} -> {output_path}", flush=True)
    # 逐行读写，避免大文件占用大量内存
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line_num, raw in enumerate(fin, 1):
            line = raw.strip()
            if not line:
                continue

            total_count += 1

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                # 保留原始行，避免破坏数据；同时给出警告
                print(f"[WARN] {input_path} 第 {line_num} 行JSON解析失败: {e}")
                fout.write(line + "\n")
                continue

            if isinstance(data, dict):
                try:
                    data, modified = _process_json_obj(data)
                    if modified:
                        processed_count += 1
                except (ValueError, AssertionError) as e:
                    print(f"[ERROR] {input_path} 第 {line_num} 行处理失败: {e}")
                    fout.write(line + "\n")
                    continue

            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    print(f"处理完成!")
    print(f"  总行数: {total_count}")
    print(f"  修改行数: {processed_count}")
    print(f"  输出文件: {output_path}")
    return total_count, processed_count


def _iter_jsonl_files(input_dir: Path, recursive: bool = True) -> Iterable[Path]:
    if recursive:
        yield from sorted(input_dir.rglob("*.jsonl"))
    else:
        yield from sorted(input_dir.glob("*.jsonl"))


def process_dir(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    workers: Optional[int] = None,
    recursive: bool = True,
) -> None:
    """
    多线程处理整个目录下的所有 jsonl 文件，并将输出写到 output_dir 下（保留相对路径结构）。
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(_iter_jsonl_files(input_dir, recursive=recursive))
    if not files:
        print(f"[WARN] 未在目录中找到任何 .jsonl: {input_dir}")
        return

    if workers is None:
        # I/O 密集型：给一个相对保守但够用的默认值
        cpu = os.cpu_count() or 8
        workers = min(32, max(4, cpu * 2))

    print(f"发现 {len(files)} 个jsonl，开始并发处理：workers={workers} recursive={recursive}")
    t0 = time.time()

    total_lines_sum = 0
    modified_lines_sum = 0

    def _job(in_path: Path) -> Tuple[Path, int, int]:
        rel = in_path.relative_to(input_dir)
        out_path = output_dir / rel
        total, modified = process_jsonl(in_path, out_path)
        return rel, total, modified

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_job, p) for p in files]
        for fut in as_completed(futs):
            rel, total, modified = fut.result()
            total_lines_sum += total
            modified_lines_sum += modified
            print(f"[OK] {rel}  总行数={total} 修改行数={modified}")

    dt = time.time() - t0
    print("目录处理完成!")
    print(f"  输入目录: {input_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  文件数量: {len(files)}")
    print(f"  总行数(累计): {total_lines_sum}")
    print(f"  修改行数(累计): {modified_lines_sum}")
    print(f"  用时: {dt:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description='检测每个 message 中是否只有一个 image_url，并给第一个 user message 的 text 开头添加 <IMG_CONTEXT>\n'
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=str(DEFAULT_INPUT_DIR),
        help=f"输入JSONL文件路径或目录（默认: {DEFAULT_INPUT_DIR}）",
    )
    parser.add_argument(
        '-o', '--output',
        default=None,
        help=f"输出JSONL文件路径或目录（目录模式默认: {DEFAULT_OUTPUT_DIR}；单文件模式默认: *_with_imgcontext.jsonl）",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=None,
        help="目录模式下的线程数（默认: min(32, max(4, 2*CPU)))",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="目录模式下不递归，只处理 input_dir 顶层的 *.jsonl（默认递归）",
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='仅在输入为单文件时生效：直接覆盖原文件'
    )
    
    args = parser.parse_args()

    input_path = Path(args.input_path)

    if input_path.is_dir():
        output_dir = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR
        process_dir(
            input_dir=input_path,
            output_dir=output_dir,
            workers=args.workers,
            recursive=not args.no_recursive,
        )
        return

    # 单文件模式（兼容旧用法）
    if args.overwrite:
        out = input_path
    else:
        out = Path(args.output) if args.output else None

    process_jsonl(input_path, out)


if __name__ == '__main__':
    main()

