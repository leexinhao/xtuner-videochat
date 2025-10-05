# coding=utf-8
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""video processor class for Qwen3-VL https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/video_processing_qwen3_vl.py"""

import math
import numpy as np
import torch

from typing import Callable, Optional, Union

from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ChannelDimension, PILImageResampling, SizeDict, get_image_size
from transformers.processing_utils import Unpack, VideosKwargs
from transformers.utils import TensorType, add_start_docstrings, logging, is_torchvision_v2_available
from transformers.video_processing_utils import BASE_VIDEO_PROCESSOR_DOCSTRING, BaseVideoProcessor
from transformers.video_utils import (
    VideoInput,
    group_videos_by_shape,
    reorder_videos,
    is_valid_video,
    make_batched_videos,
)
from .videochat3_utils import VideoChat3VideoMetadata

if is_torchvision_v2_available():
    from torchvision.transforms.v2 import functional as F
else:
    from torchvision.transforms import functional as F

logger = logging.get_logger(__name__)




def smart_video_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 1,
    factor: int = 28,
    frame_min_pixels: int = 16 * 28 * 28 * 4,
    frame_max_pixels: int = 1024 * 28 * 28 * 4,
    video_max_total_pixels: int = 5000 * 28 * 28 * 4,
):
    assert temporal_factor == 1, "temporal_factor must be 1 for videochat3!"
    if num_frames < temporal_factor:
        raise ValueError(f"t:{num_frames} must be larger than temporal_factor:{temporal_factor}")
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )

    h_bar, w_bar = smart_resize(height, width, factor, frame_min_pixels, frame_max_pixels)
    t_bar = round(num_frames / temporal_factor) * temporal_factor

    if t_bar * h_bar * w_bar > video_max_total_pixels:
        beta = math.sqrt((num_frames * height * width) / video_max_total_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)

    return h_bar, w_bar


class VideoChat3VideoProcessorInitKwargs(VideosKwargs):
    patch_size: Optional[int]
    temporal_patch_size: Optional[int]
    merge_size: Optional[int]
    min_frames: Optional[int]
    max_frames: Optional[int]


@add_start_docstrings(
    "Constructs a fast Qwen3-VL image processor that dynamically resizes videos based on the original videos.",
    BASE_VIDEO_PROCESSOR_DOCSTRING,
    """
        patch_size (`int`, *optional*, defaults to 16):
            The spacial patch size of the vision encoder.
        temporal_patch_size (`int`, *optional*, defaults to 2):
            The temporal patch size of the vision encoder.
        merge_size (`int`, *optional*, defaults to 2):
            The merge size of the vision encoder to llm encoder.
    """,
)
class VideoChat3VideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 128 * 32 * 32, "longest_edge": 768 * 32 * 32}
    video_max_total_pixels = 768 * 32 * 32
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    patch_size = 14
    temporal_patch_size = 1
    merge_size = 2
    temporal_merge_size = 4
    fps = 2
    min_frames = 4
    max_frames = 1024
    do_sample_frames = True
    valid_kwargs = VideoChat3VideoProcessorInitKwargs
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(self, **kwargs: Unpack[VideoChat3VideoProcessorInitKwargs]):
        super().__init__(**kwargs)
        if self.size is not None and (
            self.size.get("shortest_edge", None) is None or self.size.get("longest_edge", None) is None
        ):
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")

    def _further_process_kwargs(
        self,
        size: Optional[SizeDict] = None,
        **kwargs,
    ) -> dict:
        """
        Update kwargs that need further processing before being validated
        Can be overridden by subclasses to customize the processing of kwargs.
        """
        if size is not None and ("shortest_edge" not in size or "longest_edge" not in size):
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")

        return super()._further_process_kwargs(size=size, **kwargs)

    def get_num_sampled_frames(
        self,
        metadata: VideoChat3VideoMetadata,
        num_frames: Optional[int] = None,
        fps: Optional[Union[int, float]] = None
        ):
        if fps is not None and num_frames is not None:
            raise ValueError("`num_frames` and `fps` are mutually exclusive arguments, please use only one!")

        if metadata.clip_start_time is not None and metadata.clip_end_time is not None:
            total_num_frames = int((metadata.clip_end_time - metadata.clip_start_time) * metadata.fps)
        else:
            total_num_frames = metadata.total_num_frames

        sample_fps = fps if fps is not None else self.fps

        # If num_frames is not given but fps is, calculate num_frames from fps
        if num_frames is None and fps is not None:
            if metadata.fps is None:
                raise ValueError("`fps` is not provided in video metadata.")
            num_sampled_frames = int(total_num_frames / metadata.fps * sample_fps)
            num_sampled_frames = min(min(max(num_sampled_frames, self.min_frames), self.max_frames), total_num_frames)
        elif num_frames is not None:
            num_sampled_frames = min(min(max(num_frames, self.min_frames), self.max_frames), total_num_frames)
        else:
            raise ValueError("`num_frames` and `fps` are not provided for sampling frames.")
            
        return num_sampled_frames

    def sample_frames(
        self,
        metadata: VideoChat3VideoMetadata,
        num_frames: Optional[int] = None,
        fps: Optional[Union[int, float]] = None,
        **kwargs,
    ):
        """
        Default sampling function which uniformly samples the desired number of frames between 0 and total number of frames.
        If `fps` is passed along with metadata, `fps` frames per second are sampled uniformty. Arguments `num_frames`
        and `fps` are mutually exclusive.

        Args:
            video (`torch.Tensor`):
                Video that need to be sampled.
            metadata (`VideoChat3VideoMetadata`):
                Metadata of the video containing information about total duration, fps and total number of frames.
            num_frames (`int`, *optional*):
                Maximum number of frames to sample. Defaults to `self.num_frames`.
            fps (`int` or `float`, *optional*):
                Target frames to sample per second. Defaults to `self.fps`.
        Returns:
            torch.Tensor:
                Sampled video frames.
        """
        num_sampled_frames = self.get_num_sampled_frames(metadata, num_frames, fps)

        if metadata.clip_start_time is not None and metadata.clip_end_time is not None:
            indices = np.linspace(metadata.clip_start_time * metadata.fps, metadata.clip_end_time * metadata.fps, num_sampled_frames).round().astype(int)
        else:
            indices = np.linspace(0, metadata.total_num_frames - 1, num_sampled_frames).round().astype(int)

        return indices

    def _decode_and_sample_videos(
        self,
        videos: VideoInput,
        video_metadata: Union[VideoChat3VideoMetadata, dict],
        do_sample_frames: Optional[bool] = None,
        sample_indices_fn: Optional[Callable] = None,
    ) -> list["torch.Tensor"]:
        """
        Decode input videos and sample frames if needed.
        """
        videos = make_batched_videos(videos)
        
        # 自定义处理video_metadata，避免使用make_batched_metadata
        if video_metadata is None:
            video_metadata = [None] * len(videos)
        elif isinstance(video_metadata, (VideoChat3VideoMetadata, dict)):
            video_metadata = [video_metadata]
        elif isinstance(video_metadata, list):
            # 确保每个元素都是VideoChat3VideoMetadata或dict
            processed_metadata = []
            for metadata in video_metadata:
                if isinstance(metadata, dict):
                    # 如果是dict，转换为VideoChat3VideoMetadata
                    processed_metadata.append(VideoChat3VideoMetadata(**metadata))
                elif isinstance(metadata, VideoChat3VideoMetadata):
                    processed_metadata.append(metadata)
                else:
                    # 如果是其他类型，尝试转换
                    processed_metadata.append(VideoChat3VideoMetadata(**metadata.__dict__))
            video_metadata = processed_metadata

        _is_valid_video = is_valid_video(videos[0])
        # Only sample frames if an array video is passed, otherwise first decode -> then sample
        if _is_valid_video and do_sample_frames:
            sampled_videos = []
            for video, metadata in zip(videos, video_metadata):
                indices = sample_indices_fn(metadata=metadata)
                metadata.frames_indices = indices # NOTE: @Lixinhao, for _calculate_timestamps!
                sampled_videos.append(video[indices])
            videos = sampled_videos
        elif not _is_valid_video:
            if isinstance(videos[0], list):
                # Videos sometimes are passed as a list of image URLs, especially through templates
                videos = [
                    torch.stack([F.pil_to_tensor(image) for image in images], dim=0)
                    for images in self.fetch_images(videos)
                ]
                if do_sample_frames:
                    sampled_videos = []
                    for video, metadata in zip(videos, video_metadata):
                        indices = sample_indices_fn(metadata=metadata)
                        metadata.frames_indices = indices # NOTE: @Lixinhao, for _calculate_timestamps!
                        sampled_videos.append(video[indices])
                    videos = sampled_videos
                else:
                    videos = [
                        torch.stack([F.pil_to_tensor(image) for image in images], dim=0)
                        for images in self.fetch_images(videos)
                    ]
            else:
                # 使用父类的fetch_videos方法，但不传递sample_indices_fn
                videos, metadata_list = super().fetch_videos(videos, sample_indices_fn=None)
                # 将VideoMetadata转换为VideoChat3VideoMetadata
                video_metadata = []
                for metadata in metadata_list:
                    if metadata is None:
                        # 如果metadata是None，跳过
                        continue
                    elif isinstance(metadata, VideoChat3VideoMetadata):
                        video_metadata.append(metadata)
                    else:
                        # 转换为VideoChat3VideoMetadata
                        video_metadata.append(VideoChat3VideoMetadata(
                            total_num_frames=metadata.total_num_frames,
                            fps=metadata.fps,
                            width=metadata.width,
                            height=metadata.height,
                            duration=metadata.duration,
                            video_backend=metadata.video_backend,
                            frames_indices=metadata.frames_indices,
                            video_start_time=0.0,
                            clip_start_time=None,
                            clip_end_time=None
                        ))
                
                # 如果需要采样帧，使用我们自己的sample_indices_fn
                if do_sample_frames and sample_indices_fn is not None:
                    sampled_videos = []
                    for video, metadata in zip(videos, video_metadata):
                        indices = sample_indices_fn(metadata=metadata)
                        metadata.frames_indices = indices
                        sampled_videos.append(video[indices])
                    videos = sampled_videos

        return videos, video_metadata

    def _preprocess(
        self,
        videos: list[torch.Tensor],
        do_convert_rgb: bool = True,
        do_resize: bool = True,
        size: Optional[SizeDict] = None,
        interpolation: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ):
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}

        for shape, stacked_videos in grouped_videos.items():
            B, T, C, H, W = stacked_videos.shape
            num_frames, height, width = T, H, W
            if do_resize:
                resized_height, resized_width = smart_video_resize(
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    temporal_factor=temporal_patch_size,
                    factor=patch_size * merge_size,
                    frame_min_pixels=size.shortest_edge,
                    frame_max_pixels=size.longest_edge,
                    video_max_total_pixels=self.video_max_total_pixels,
                )
                stacked_videos = stacked_videos.view(B * T, C, H, W)
                stacked_videos = self.resize(
                    stacked_videos,
                    size=SizeDict(height=resized_height, width=resized_width),
                    interpolation=interpolation,
                )
                stacked_videos = stacked_videos.view(B, T, C, resized_height, resized_width)
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        # Group videos by size for further processing
        # Needed in case do_resize is False, or resize returns videos with different sizes
        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_grids = {}
        for shape, stacked_videos in grouped_videos.items():
            resized_height, resized_width = get_image_size(stacked_videos[0], channel_dim=ChannelDimension.FIRST)

            # Fused rescale and normalize
            stacked_videos = self.rescale_and_normalize(
                stacked_videos, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            patches = stacked_videos

            # Check that videos have `num_frames` divisible by `temporal_patch_size`
            if patches.shape[1] % temporal_patch_size != 0:
                repeats = patches[:, -1:].repeat(1, temporal_patch_size - 1, 1, 1, 1)
                patches = torch.cat([patches, repeats], dim=1)
            batch_size, grid_t, channel = patches.shape[:3]
            grid_t = grid_t // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.view(
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )

            processed_videos_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_grids = reorder_videos(processed_grids, grouped_videos_index)
        pixel_values_videos = torch.cat(processed_videos, dim=0)
        video_grid_thw = torch.tensor(processed_grids)
        data = {
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }

        return BatchFeature(data=data, tensor_type=return_tensors)

    def get_number_of_video_tokens(self, num_frames: int, height: int, width: int, videos_kwargs=None):
        if num_frames % self.temporal_merge_size != 0:
            num_clips = num_frames // self.temporal_merge_size + 1
        else:
            num_clips = num_frames // self.temporal_merge_size
        return num_clips * height * width // self.merge_size**2

__all__ = ["VideoChat3VideoProcessor"]