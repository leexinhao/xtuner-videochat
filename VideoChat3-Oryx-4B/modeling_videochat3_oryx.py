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
VideoChat3-Oryx: VideoChat3 with Oryx SigLIP-based vision encoder (no RoPE).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling, ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.utils.import_utils import is_flash_attn_2_available
from transformers import AutoModel
from .configuration_videochat3_oryx import VideoChat3OryxConfig, VideoChat3OryxVisionConfig


if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func
else:
    flash_attn_varlen_func = None


# ---------------------------------------------------------------------------
# Positional embedding (reused from VideoChat3: spatial bicubic + temporal sincos)
# ---------------------------------------------------------------------------

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


class VideoChat3InterpPosEmb(nn.Module):
    def __init__(
        self, height: int, width: int, max_clip_length: int, dim: int, interpolation_mode: str = "bicubic"
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.max_clip_length = max_clip_length
        self.interpolation_mode = interpolation_mode
        self.weight = nn.Parameter(torch.empty(height, width, dim))
        self.time_weight = nn.Parameter(torch.empty(max_clip_length, 1, dim))
        self.dim = dim
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)
        initial_time_weight = (
            torch.from_numpy(get_1d_sincos_pos_embed_from_grid(self.dim, np.arange(self.max_clip_length, dtype=np.float32)))
            .float()
            .unsqueeze(1)
        )
        with torch.no_grad():
            self.time_weight.copy_(initial_time_weight)

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        pos_embs = []
        for t, h, w in grid_thws.tolist():
            if (h, w) == self.weight.shape[:-1]:
                pos_emb_2d = self.weight.flatten(end_dim=1)
            else:
                pos_emb_2d = (
                    F.interpolate(
                        self.weight.permute((2, 0, 1)).unsqueeze(0),
                        size=(h, w),
                        mode=self.interpolation_mode,
                    )
                    .squeeze(0)
                    .permute((1, 2, 0))
                    .flatten(end_dim=1)
                )

            if t == 1:
                pos_emb_3d = pos_emb_2d
            else:
                pos_emb_3d = pos_emb_2d.unsqueeze(0).repeat(t, 1, 1) + self.time_weight[:t]

            pos_embs.append(pos_emb_3d.reshape(-1, pos_emb_3d.shape[-1]))

        out = x + torch.cat(pos_embs)
        return out


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------

class VideoChat3OryxVisionPatchEmbed(nn.Module):
    def __init__(
        self,
        out_dim: int,
        in_dim: int = 3,
        patch_size: Union[int, tuple[int, int]] = (16, 16),
        pos_emb_height: int = 128,
        pos_emb_width: int = 128,
        max_clip_length: int = 4,
    ):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        assert len(patch_size) == 2
        self.patch_size = patch_size
        self.in_dim = in_dim
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=patch_size, stride=patch_size)

        self.pos_emb = VideoChat3InterpPosEmb(
            height=pos_emb_height, width=pos_emb_width, max_clip_length=max_clip_length, dim=out_dim
        )

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.in_dim, self.patch_size[0], self.patch_size[1])
        x = self.proj(x).view(x.size(0), -1)
        x = self.pos_emb(x, grid_thws)
        return x


# ---------------------------------------------------------------------------
# Attention (no RoPE — key difference from VideoChat3)
# ---------------------------------------------------------------------------

def flash_attention_2(q, k, v, q_cu_seqlens=None, k_cu_seqlens=None):
    assert q.dim() == k.dim() == v.dim() == 3
    assert q_cu_seqlens[-1] == q.shape[0]
    assert k_cu_seqlens[-1] == k.shape[0] == v.shape[0]

    max_seqlen_q = (q_cu_seqlens[1:] - q_cu_seqlens[:-1]).max().item()
    max_seqlen_k = (k_cu_seqlens[1:] - k_cu_seqlens[:-1]).max().item()
    attn_out = flash_attn_varlen_func(
        q, k, v,
        q_cu_seqlens, k_cu_seqlens,
        max_seqlen_q, max_seqlen_k,
        causal=False,
    )
    attn_out = attn_out.flatten(start_dim=-2)
    return attn_out


def eager_attention(q, k, v, q_cu_seqlens=None, k_cu_seqlens=None):
    seq_length = q.shape[0]
    attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
    for i in range(1, len(q_cu_seqlens)):
        attention_mask[
            ...,
            q_cu_seqlens[i - 1] : q_cu_seqlens[i],
            q_cu_seqlens[i - 1] : q_cu_seqlens[i],
        ] = True
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)

    attn_weight = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    attn_weight += attention_mask
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32).to(q.dtype)

    attn_output = attn_weight @ v
    attn_output = attn_output.transpose(0, 1)
    attn_output = attn_output.reshape(seq_length, -1)
    return attn_output


VL_VISION_ATTENTION_FUNCTIONS = {
    "flash_attention_2": flash_attention_2,
    "sdpa": eager_attention,
    "eager": eager_attention,
}


# ---------------------------------------------------------------------------
# Vision transformer layer (no RoPE)
# ---------------------------------------------------------------------------

class VideoChat3OryxVisionMLP(nn.Module):
    def __init__(self, dims: list[int], activation, bias=True):
        super().__init__()
        assert len(dims) == 3
        self.fc1 = nn.Linear(dims[0], dims[1], bias=bias)
        self.fc2 = nn.Linear(dims[1], dims[2], bias=bias)
        self.activation = activation
        for m in [self.fc1, self.fc2]:
            nn.init.trunc_normal_(m.weight, std=math.sqrt(2 / m.in_features))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        return self.fc2(x)


class VideoChat3OryxVisionLayer(GradientCheckpointingLayer):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        *,
        attn_impl: str = "eager",
        activation=F.gelu,
        attn_bias: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.hidden_size_per_attention_head = hidden_dim // num_heads
        self.attn_impl = attn_impl

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = VideoChat3OryxVisionMLP([hidden_dim, mlp_dim, hidden_dim], activation)
        self.attn = nn.ModuleDict({
            "qkv": nn.Linear(hidden_dim, hidden_dim * 3, bias=attn_bias),
            "proj": nn.Linear(hidden_dim, hidden_dim, bias=attn_bias),
        })

    def attention_forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor):
        xqkv = self.attn["qkv"](x)
        qkv_shape = xqkv.size()[:-1] + (3, self.num_heads, self.hidden_size_per_attention_head)
        xqkv = xqkv.view(*qkv_shape)
        xq, xk, xv = torch.unbind(xqkv, dim=-3)

        attn_func = VL_VISION_ATTENTION_FUNCTIONS[self.attn_impl]
        attn_out = attn_func(xq, xk, xv, q_cu_seqlens=cu_seqlens, k_cu_seqlens=cu_seqlens)

        attn_out = self.attn["proj"](attn_out)
        return attn_out

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        attn_out = self.attention_forward(hidden_states, cu_seqlens)
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.mlp(self.norm2(hidden_states))
        hidden_states = residual + hidden_states
        return hidden_states


# ---------------------------------------------------------------------------
# Vision encoder (no RoPE, no final layernorm)
# ---------------------------------------------------------------------------

class VideoChat3OryxVisionEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, block_cfg: dict) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([VideoChat3OryxVisionLayer(**block_cfg) for _ in range(num_layers)])

    def forward(self, hidden_states: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        lengths = torch.cat(
            (
                torch.zeros(1, device=hidden_states.device, dtype=grid_thws.dtype),
                grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2],
            )
        )
        cu_seqlens = lengths.cumsum(dim=0, dtype=torch.int32)

        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens)

        return hidden_states


# ---------------------------------------------------------------------------
# Patch merger (identical to VideoChat3)
# ---------------------------------------------------------------------------

def patch_merger(x, grid_thws, merge_kernel_size=(2, 2)):
    d_model = x.size(-1)
    outputs = []
    pre_sum = 0
    for t, h, w in grid_thws.tolist():
        seq = x[pre_sum : pre_sum + t * h * w]
        kernel_height, kernel_width = merge_kernel_size
        new_height, new_width = h // kernel_height, w // kernel_width
        reshaped_seq = seq.view(t, new_height, kernel_height, new_width, kernel_width, d_model)
        reshaped_seq = reshaped_seq.permute(0, 1, 3, 2, 4, 5).contiguous().mean(dim=0)
        padded_seq = reshaped_seq.view(new_height * new_width, kernel_height * kernel_width, -1)
        outputs.append(padded_seq)
        pre_sum += t * h * w
    return outputs


# ---------------------------------------------------------------------------
# Vision model
# ---------------------------------------------------------------------------

class VideoChat3OryxVisionPreTrainedModel(PreTrainedModel):
    config_class = VideoChat3OryxVisionConfig
    base_model_prefix = "videochat3_oryx_vision"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["VideoChat3OryxVisionLayer"]
    _supports_sdpa = True
    _supports_flash_attn = True


class VideoChat3OryxVisionModel(VideoChat3OryxVisionPreTrainedModel):
    def __init__(self, config: VideoChat3OryxVisionConfig) -> None:
        super().__init__(config)
        self.config = config

        self.patch_embed = VideoChat3OryxVisionPatchEmbed(
            out_dim=config.hidden_size,
            patch_size=config.patch_size,
            pos_emb_height=config.init_pos_emb_height,
            pos_emb_width=config.init_pos_emb_width,
            max_clip_length=config.temporal_merge_size,
        )
        self.encoder = VideoChat3OryxVisionEncoder(
            hidden_dim=config.hidden_size,
            num_layers=config.num_hidden_layers,
            block_cfg={
                "num_heads": config.num_attention_heads,
                "hidden_dim": config.hidden_size,
                "mlp_dim": config.intermediate_size,
                "activation": ACT2FN["gelu"],
                "attn_bias": True,
                "attn_impl": config.attn_impl,
            },
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.patch_embed.pos_emb

    def split_grid_thws_clip_by_clip(self, grid_thws: torch.Tensor) -> torch.Tensor:
        tmp_thw_list = []
        for t, h, w in grid_thws.tolist():
            if t > self.config.temporal_merge_size:
                _t = t
                for _ in range(self.config.temporal_merge_size, t, self.config.temporal_merge_size):
                    tmp_thw_list.append([self.config.temporal_merge_size, h, w])
                    _t -= self.config.temporal_merge_size
                if _t != 0:
                    tmp_thw_list.append([_t, h, w])
            else:
                assert t != 0, grid_thws
                tmp_thw_list.append([t, h, w])
        return torch.tensor(tmp_thw_list, device=grid_thws.device, dtype=grid_thws.dtype)

    def forward(self, pixel_values: torch.Tensor, grid_thws: torch.Tensor) -> list[torch.Tensor]:
        grid_thws = self.split_grid_thws_clip_by_clip(grid_thws)
        hidden_states = self.patch_embed(pixel_values, grid_thws)
        hidden_states = self.encoder(hidden_states, grid_thws)
        hidden_states = patch_merger(hidden_states, grid_thws, merge_kernel_size=self.config.merge_kernel_size)
        return hidden_states


# ---------------------------------------------------------------------------
# Multi-modal projector (identical to VideoChat3)
# ---------------------------------------------------------------------------

class VideoChat3OryxMultiModalProjector(nn.Module):
    def __init__(self, config: VideoChat3OryxConfig):
        super().__init__()
        self.config = config
        vision_hidden_size = config.vision_config.hidden_size
        merge_kernel_size = config.vision_config.merge_kernel_size
        self.hidden_size = vision_hidden_size * merge_kernel_size[0] * merge_kernel_size[1]

        text_hidden_size = getattr(config.text_config, "hidden_size", 2048)

        self.pre_norm = nn.LayerNorm(vision_hidden_size, eps=1e-05)
        self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(self.hidden_size, text_hidden_size, bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        if isinstance(image_features, list):
            image_features = torch.cat(image_features, dim=0)
        hidden_states = self.pre_norm(image_features).view(-1, self.hidden_size)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Full model outputs
# ---------------------------------------------------------------------------

@dataclass
class VideoChat3OryxModelOutputWithPast(BaseModelOutputWithPast):
    image_hidden_states: Optional[torch.FloatTensor] = None
    video_hidden_states: Optional[torch.FloatTensor] = None


@dataclass
class VideoChat3OryxCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None
    video_hidden_states: Optional[torch.FloatTensor] = None


# ---------------------------------------------------------------------------
# Full multimodal model
# ---------------------------------------------------------------------------

class VideoChat3OryxPreTrainedModel(PreTrainedModel):
    config_class = VideoChat3OryxConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True


class VideoChat3OryxModel(VideoChat3OryxPreTrainedModel):
    _checkpoint_conversion_mapping = {}

    def __init__(self, config: VideoChat3OryxConfig):
        super().__init__(config)
        self.vision_tower = VideoChat3OryxVisionModel._from_config(config.vision_config)
        self.multi_modal_projector = VideoChat3OryxMultiModalProjector(config)
        self.language_model = AutoModel.from_config(config.text_config, trust_remote_code=True)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_image_features(self, pixel_values, grid_thws, **kwargs):
        pixel_values = pixel_values.to(dtype=self.dtype)
        vision_features = self.vision_tower(pixel_values=pixel_values, grid_thws=grid_thws)
        vision_features = self.multi_modal_projector(vision_features)
        return vision_features

    def get_video_features(self, pixel_values_videos, video_grid_thw=None):
        return self.get_image_features(pixel_values_videos, video_grid_thw)

    def get_placeholder_mask(self, input_ids, inputs_embeds, image_features=None, video_features=None):
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = (input_ids == self.config.image_token_id)
            special_video_mask = (input_ids == self.config.video_token_id)

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape}"
            )

        return special_image_mask, special_video_mask

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, VideoChat3OryxModelOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_embeds = None
        video_embeds = None

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        return VideoChat3OryxModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_embeds if pixel_values is not None else None,
            video_hidden_states=video_embeds if pixel_values_videos is not None else None,
        )


class VideoChat3OryxForConditionalGeneration(VideoChat3OryxPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: VideoChat3OryxConfig):
        super().__init__(config)
        self.model = VideoChat3OryxModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_image_features(self, pixel_values, image_grid_thw, **kwargs):
        return self.model.get_image_features(pixel_values=pixel_values, grid_thws=image_grid_thw, **kwargs)

    def get_video_features(self, pixel_values_videos, video_grid_thw, **kwargs):
        return self.model.get_video_features(pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw, **kwargs)

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def vision_tower(self):
        return self.model.vision_tower

    @property
    def multi_modal_projector(self):
        return self.model.multi_modal_projector

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, VideoChat3OryxCausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return VideoChat3OryxCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
            video_hidden_states=outputs.video_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            **kwargs,
        )

        is_decoding_step = (
            (model_inputs["inputs_embeds"] is not None and model_inputs["inputs_embeds"].shape[1] == 1)
            or (model_inputs["input_ids"] is not None and model_inputs["input_ids"].shape[1] == 1)
        )
        if cache_position[0] != 0 and is_decoding_step:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs


__all__ = [
    "VideoChat3OryxVisionPreTrainedModel",
    "VideoChat3OryxVisionModel",
    "VideoChat3OryxPreTrainedModel",
    "VideoChat3OryxModel",
    "VideoChat3OryxForConditionalGeneration",
]
