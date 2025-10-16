from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

from mmengine import is_installed
from pydantic import BaseModel, ConfigDict
from typing_extensions import Self

from xtuner.v1.float8 import Float8Config
from xtuner.v1.model.dense.qwen3 import Qwen3Dense8BConfig
from xtuner.v1.model.moe.moe import MoEConfig, TransformerConfig
from xtuner.v1.model.moe.qwen3 import Qwen3MoE235BA22Config
from xtuner.v1.utils import get_logger


if TYPE_CHECKING:
    from .modeling_videochat3 import VideoChat3ForConditionalGeneration

logger = get_logger()


class VideoChat3VisionConfig(BaseModel):
    model_config = ConfigDict(
        title="VideoChat3 vision config for xtuner",
        extra="allow",
    )
    # 基于VideoChat3-debug/config.json的vision_config
    model_type: str = "moonvit"
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_attention_heads: int = 16
    num_hidden_layers: int = 27
    hidden_act: str = "gelu"
    patch_size: int = 14  # 从16改为14
    merge_kernel_size: list[int] = [2, 2]  # 新增
    temporal_patch_size: int = 1  # 从2改为1
    temporal_merge_size: int = 4  # 新增
    init_pos_emb_height: int = 64  # 新增
    init_pos_emb_width: int = 64  # 新增
    in_channels: int = 3
    initializer_range: float = 0.02
    torch_dtype: str = "bfloat16"  # 新增
    float8_cfg: Optional["Float8Config"] = None
    attn_impl: Literal["flash_attention", "flex_attention", "eager_attention"] = "eager_attention"
    _attn_implementation: str = "eager"  # 新增，用于VideoChat3VisionLayer

    def model_post_init(self, _):
        if not is_installed("flash-attn") and self.attn_impl == "flash_attention":
            logger.warning("flash-attn is not installed, using `flex_attention` instead.")
            self.attn_impl = "flex_attention"
        return self

    def build(self):
        from .modeling_vision import VideoChat3VisionModel

        return VideoChat3VisionModel(self)


class VideoChat3ProjectorConfig(BaseModel):
    # 基于KimiVLMultiModalProjector的配置
    vision_hidden_size: int = 1152
    text_hidden_size: int = 2048
    merge_kernel_size: list[int] = [2, 2]  # 与vision_config保持一致
    float8_cfg: Optional["Float8Config"] = None

    def build(self):
        from .modeling_projector import VideoChat3MultiModalProjector

        return VideoChat3MultiModalProjector(self)


class VideoChat3BaseConfig(BaseModel):
    model_config = ConfigDict(
        title="Base VideoChat3 model config for xtuner",
        extra="allow",
    )
    vision_config: VideoChat3VisionConfig
    projector_config: VideoChat3ProjectorConfig
    text_config: TransformerConfig

    # 基于VideoChat3-debug/config.json
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    
    # 基于video_preprocessor_config.json和config.json
    patch_size: int = 14
    temporal_patch_size: int = 1
    merge_size: int = 2
    temporal_merge_size: int = 4
    merge_kernel_size: list[int] = [2, 2]  # 与vision_config保持一致
    
    freeze_vision: bool = False
    freeze_projector: bool = False
    freeze_language: bool = False

    def build(self) -> "VideoChat3ForConditionalGeneration":
        from .modeling_videochat3 import VideoChat3ForConditionalGeneration

        return VideoChat3ForConditionalGeneration(self)

    @classmethod
    def from_hf(cls, hf_path: str | Path) -> Self:
        raise NotImplementedError


class VideoChat3MoEConfig(VideoChat3BaseConfig):
    # 基于VideoChat3-debug/config.json的完整配置
    vision_config: VideoChat3VisionConfig = VideoChat3VisionConfig()
    projector_config: VideoChat3ProjectorConfig = VideoChat3ProjectorConfig()
    text_config: MoEConfig = Qwen3MoE235BA22Config(
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=6144,
        num_attention_heads=32,
        num_hidden_layers=48,
        num_experts=128,
        num_experts_per_tok=8,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=262144,
        rms_norm_eps=1e-06,
        attention_bias=False,
        attention_dropout=0.0,
        initializer_range=0.02,
        head_dim=128,
        moe_intermediate_size=768,
        norm_topk_prob=True,
        mlp_only_layers=[],
        rope_theta=5000000,
        use_cache=True,
        rope_scaling={
            "rope_type": "default",
            "mrope_interleaved": True,
            "mrope_section": [24, 20, 20]
        }
    )

    @property
    def hf_config(self):
        # TODO(pppppM) Support saving HuggingFace format config
        logger.warning(
            f"{type(self)} does not support conversion to HuggingFace config format. "
            "Only the original HuggingFace config will be retained in the saved HuggingFace format checkpoint. "
            f"If you have changed the default values in {type(self)}, it may cause the config in the saved "
            "HuggingFace format checkpoint to not match the weights."
        )
        return None


class VideoChat3DenseConfig(VideoChat3BaseConfig):
    # 简化版本，使用dense模型而不是MoE
    vision_config: VideoChat3VisionConfig = VideoChat3VisionConfig()
    projector_config: VideoChat3ProjectorConfig = VideoChat3ProjectorConfig()
    text_config: Qwen3Dense8BConfig = Qwen3Dense8BConfig(vocab_size=151936)

    @property
    def hf_config(self):
        # TODO(pppppM) Support saving HuggingFace format config
        logger.warning(
            f"{type(self)} does not support conversion to HuggingFace config format. "
            "Only the original HuggingFace config will be retained in the saved HuggingFace format checkpoint. "
            f"If you have changed the default values in {type(self)}, it may cause the config in the saved "
            "HuggingFace format checkpoint to not match the weights."
        )
        return None
