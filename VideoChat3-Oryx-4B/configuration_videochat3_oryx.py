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

from typing import Any, Optional

from transformers.configuration_utils import PretrainedConfig
from transformers import CONFIG_MAPPING, AutoConfig


class VideoChat3OryxVisionConfig(PretrainedConfig):
    """Configuration for VideoChat3-Oryx vision encoder (SigLIP-based ViT)."""

    model_type = "videochat3_oryx_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size: int = 1152,
        num_hidden_layers: int = 27,
        num_attention_heads: int = 16,
        patch_size: int = 16,
        mlp_ratio: float = 3.7362,
        merge_kernel_size: tuple[int, int] = (2, 2),
        temporal_patch_size: int = 1,
        temporal_merge_size: int = 4,
        init_pos_emb_height: int = 128,
        init_pos_emb_width: int = 128,
        dtype: str = "bfloat16",
        attn_impl: str = "flash_attention_2",
        **kwargs,
    ):
        kwargs.pop("torch_dtype", None)
        super().__init__(**kwargs)

        if merge_kernel_size is None:
            merge_kernel_size = [2, 2]

        self.hidden_size = hidden_size
        self.intermediate_size = int(hidden_size * mlp_ratio)
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.patch_size = patch_size
        self.mlp_ratio = mlp_ratio
        self.merge_kernel_size = merge_kernel_size
        self.temporal_patch_size = temporal_patch_size
        self.temporal_merge_size = temporal_merge_size
        self.init_pos_emb_height = init_pos_emb_height
        self.init_pos_emb_width = init_pos_emb_width
        self.dtype = dtype
        self.attn_impl = attn_impl


class VideoChat3OryxConfig(PretrainedConfig):
    """Configuration for the full VideoChat3-Oryx multimodal model."""

    model_type = "videochat3_oryx"
    sub_configs = {"text_config": AutoConfig, "vision_config": VideoChat3OryxVisionConfig}

    def __init__(
        self,
        vision_config: Optional[dict[str, Any]] = None,
        text_config: Optional[dict[str, Any]] = None,
        image_token_id: int = 151655,
        video_token_id: int = 151656,
        vision_start_token_id: int = 151652,
        vision_end_token_id: int = 151653,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

        if isinstance(vision_config, dict):
            self.vision_config = VideoChat3OryxVisionConfig(**vision_config)
        elif isinstance(vision_config, VideoChat3OryxVisionConfig):
            self.vision_config = vision_config
        elif vision_config is None:
            self.vision_config = VideoChat3OryxVisionConfig()

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "qwen2")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["qwen2"]()

        self.text_config = text_config

        super().__init__(**kwargs)
