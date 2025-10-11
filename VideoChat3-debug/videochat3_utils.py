from transformers.video_utils import VideoMetadata

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