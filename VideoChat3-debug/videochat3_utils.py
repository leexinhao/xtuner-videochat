from dataclasses import dataclass, fields
from typing import Optional, Mapping

@dataclass
class VideoChat3VideoMetadata(Mapping):
    total_num_frames: int
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    duration: Optional[float] = None
    video_backend: Optional[str] = None
    frames_indices: Optional[list[int]] = None
    video_start_time: float = 0.0 # The start time of the video, in seconds
    clip_start_time: Optional[float] = None # The start time of the video clip to be extracted, in seconds
    clip_end_time: Optional[float] = None # The end time of the video clip to be extracted, in seconds

    def __post_init__(self):
        if self.fps is not None and self.duration is not None:
            expected_frames = self.fps * self.duration
            if abs(expected_frames - self.total_num_frames) > 1e-6:
                raise ValueError(f"fps * duration must be equal to total_num_frames, but got {expected_frames} != {self.total_num_frames}")
            
        if self.video_start_time < 0 or (self.duration is not None and self.video_start_time >= self.duration):
            raise ValueError(f"video_start_time must be greater than or equal to 0 and less than duration, but got {self.video_start_time}")

        if (self.clip_start_time is None) != (self.clip_end_time is None):
            raise ValueError("clip_start_time and clip_end_time must both be None or both be not None.")
        if self.clip_start_time is not None and self.clip_end_time is not None and self.clip_end_time <= self.clip_start_time:
            raise ValueError(f"clip_end_time must be greater than clip_start_time, but got {self.clip_end_time} <= {self.clip_start_time}")
        if self.clip_start_time is not None and self.clip_start_time < 0:
            raise ValueError(f"clip_start_time must be greater than 0, but got {self.clip_start_time}")
        if self.clip_end_time is not None and (self.clip_end_time < 0 or self.clip_end_time > self.duration):
            raise ValueError(f"clip_end_time must be greater than 0 and less than duration, but got {self.clip_end_time}")

    def __iter__(self):
        return (f.name for f in fields(self))

    def __len__(self):
        return len(fields(self))

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    @property
    def timestamps(self) -> float:
        "Timestamps of the sampled frames in seconds."
        if self.fps is None:
            raise ValueError("Cannot infer video `timestamps` when `fps` is None.")
        return [self.video_start_time + frame_idx / self.fps for frame_idx in self.frames_indices]

    def update(self, dictionary):
        for key, value in dictionary.items():
            if hasattr(self, key):
                setattr(self, key, value)