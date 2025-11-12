import os
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def extract_frames_ffmpeg(video_path, input_dir, output_dir, fps=1):
    """使用FFmpeg从视频中按指定帧率提取帧，保持原始目录结构"""
    # 获取视频的相对路径（相对于输入目录）
    rel_path = os.path.relpath(video_path, input_dir)
    rel_dir, video_name = os.path.split(rel_path)
    video_base, video_ext = os.path.splitext(video_name)

    # 创建对应的输出目录（使用视频文件名作为子目录）
    video_output_dir = os.path.join(output_dir, rel_dir, video_base)

    # 检查目标文件夹是否已经存在且包含帧文件
    if os.path.exists(video_output_dir):
        frame_count = len(os.listdir(video_output_dir))
        print(f"跳过 {rel_path}: 目标文件夹已存在，包含 {frame_count} 帧")
        return frame_count

    # 如果文件夹不存在或为空，则创建文件夹并提取帧
    os.makedirs(video_output_dir, exist_ok=True)

    # 构建FFmpeg命令
    output_pattern = os.path.join(video_output_dir, f"{video_base}_frame_%06d.jpg")
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-r", str(fps),  # 设置输出帧率
        "-q:v", "2",     # 设置JPEG质量 (1-31, 1为最佳)
        "-y",            # 覆盖已存在文件
        output_pattern
    ]

    try:
        # 执行FFmpeg命令
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True
        )
        # 获取输出的帧数量
        frame_count = len([f for f in os.listdir(video_output_dir) if f.endswith('.jpg')])
        print(f"已从 {rel_path} 提取 {frame_count} 帧，保存在 {os.path.relpath(video_output_dir, output_dir)}")
        return frame_count
    except subprocess.CalledProcessError as e:
        print(f"错误处理 {rel_path}: {e.stderr}")
        return 0

def main():
    parser = argparse.ArgumentParser(description='视频按指定帧率拆帧工具 (支持多级目录)')
    parser.add_argument('--input', '-i', required=True, help='输入视频文件夹路径')
    parser.add_argument('--output', '-o', default='./frames_output', help='输出文件夹路径')
    parser.add_argument('--fps', '-f', type=int, default=1, help='目标帧率 (默认: 1fps)')
    parser.add_argument('--workers', '-w', type=int, default=0, help='并行工作进程数 (默认: CPU核心数)')
    args = parser.parse_args()

    # 检查输入目录是否存在
    input_dir = os.path.abspath(args.input)
    if not os.path.isdir(input_dir):
        print(f"错误: 输入目录 {input_dir} 不存在")
        return

    # 创建输出目录
    output_dir = os.path.abspath(args.output + f'_fps{args.fps}')
    os.makedirs(output_dir, exist_ok=True)

    # 支持的视频格式
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.avi']
    no_video_extensions = ['.metadata', '.lock', '.zip', '.zst', '.tar', '.frame_meta', '.json', '.jsonl', 'md', '.gitattributes', '.gitignore']
    # 获取所有视频文件（保留完整路径）
    video_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))
            elif not any(file.lower().endswith(ext) for ext in no_video_extensions):
                print(file)
                # raise ValueError(file)

    if not video_files:
        print(f"错误: 在 {input_dir} 中未找到视频文件")
        return

    video_files = sorted(video_files)
    # 确定工作进程数
    workers = args.workers if args.workers > 0 else os.cpu_count()
    print(f"找到 {len(video_files)} 个视频文件，使用 {workers} 个工作进程")

    # 使用进程池并行处理视频
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # 创建带有固定参数的partial函数
        process_video = partial(
            extract_frames_ffmpeg, 
            input_dir=input_dir,
            output_dir=output_dir, 
            fps=args.fps
        )

        # 执行并行处理
        results = list(tqdm(
            executor.map(process_video, video_files),
            total=len(video_files),
            desc="处理视频"
        ))

    total_frames = sum(results)
    print(f"所有视频处理完成，共提取 {total_frames} 帧，保存在 {output_dir}")

if __name__ == "__main__":
    main()