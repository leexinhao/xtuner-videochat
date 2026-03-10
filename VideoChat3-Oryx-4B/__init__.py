# coding=utf-8
# Copyright 2025 The VideoChat3 Team and HuggingFace Inc. team. All rights reserved.
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

"""
VideoChat3-Oryx model: VideoChat3 with Oryx SigLIP-based vision encoder.
"""

from configuration_videochat3_oryx import VideoChat3OryxConfig, VideoChat3OryxVisionConfig
from modeling_videochat3_oryx import (
    VideoChat3OryxForConditionalGeneration,
    VideoChat3OryxMultiModalProjector,
    VideoChat3OryxVisionModel,
)
from processing_videochat3 import VideoChat3Processor
from video_processing_videochat3 import VideoChat3VideoProcessor

__all__ = [
    "VideoChat3OryxConfig",
    "VideoChat3OryxVisionConfig",
    "VideoChat3OryxForConditionalGeneration",
    "VideoChat3OryxVisionModel",
    "VideoChat3OryxMultiModalProjector",
    "VideoChat3Processor",
    "VideoChat3VideoProcessor",
]
