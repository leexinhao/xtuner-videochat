import io
import random

import numpy as np
from PIL import Image
from transformers.video_utils import VideoMetadata

try:
    from decord import VideoReader
except Exception:
    VideoReader = None


class VideoChat3VideoMetadata(VideoMetadata):
    video_start_time: float = 0.0 # The start time of the video, in seconds
    clip_start_time: float = None # The start time of the video clip to be extracted, in seconds
    clip_end_time: float = None # The end time of the video clip to be extracted, in seconds

    def __post_init__(self):
        if self.fps * self.duration != self.total_num_frames:
            raise ValueError(f"fps * duration must be equal to total_num_frames, but got {self.fps * self.duration} != {self.total_num_frames}")
            
        if self.video_start_time <= 0 or self.video_start_time is None or self.video_start_time >= self.duration:
            raise ValueError(f"video_start_time must be greater than 0 and less than duration, but got {self.video_start_time}")

        if (self.clip_start_time is None) != (self.clip_end_time is None):
            raise ValueError("clip_start_time and clip_end_time must both be None or both be not None.")
        if self.clip_start_time is not None and self.clip_end_time is not None and self.clip_end_time <= self.clip_start_time:
            raise ValueError(f"clip_end_time must be greater than clip_start_time, but got {self.clip_end_time} <= {self.clip_start_time}")
        if self.clip_start_time is not None and self.clip_start_time <= 0:
            raise ValueError(f"clip_start_time must be greater than 0, but got {self.clip_start_time}")
        if self.clip_end_time is not None and self.clip_end_time <= 0 or self.clip_end_time >= self.duration:
            raise ValueError(f"clip_end_time must be greater than 0 and less than duration, but got {self.clip_end_time}")

    @property
    def timestamps(self) -> float:
        "Timestamps of the sampled frames in seconds."
        if self.fps is None:
            raise ValueError("Cannot infer video `timestamps` when `fps` is None.")
        return [self.video_start_time + frame_idx / self.fps for frame_idx in self.frames_indices]


def get_frame_indices(num_frames, vlen, sample="rand", fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ["rand", "middle"]:  # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == "rand":
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except Exception:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == "middle":
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[: len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices


def read_frames_decord(
    video_path,
    num_frames,
    sample="rand",
    fix_start=None,
    client=None,
    clip=None,
    min_num_frames=4,
    random_frame_num=None,
):
    assert VideoReader is not None, "Please install decord: pip install decord"
    if "s3://" in video_path:
        video_bytes = client.get(video_path)
        video_reader = VideoReader(io.BytesIO(video_bytes), num_threads=1)
    else:
        video_reader = VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)
    if clip:
        start, end = clip
        duration = end - start
        vlen = int(duration * fps)
        start_index = int(start * fps)

    # t_num_frames = min(max(int(duration * sample_fps), min_num_frames), num_frames)
    if random_frame_num is None:
        t_num_frames = np.random.randint(min_num_frames, num_frames + 1)
    else:
        t_num_frames = random_frame_num

    frame_indices = get_frame_indices(t_num_frames, vlen, sample=sample, fix_start=fix_start, input_fps=fps)
    if clip:
        frame_indices = [f + start_index for f in frame_indices]
    frames = video_reader.get_batch(frame_indices).asnumpy()  # (T, H, W, C), np.uint8
    frames = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]
    return frames
