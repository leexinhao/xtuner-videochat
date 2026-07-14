import os

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.datasets import DataloaderConfig, DatasetConfig, Qwen3VLTokenizeFnConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.model.compose.qwen3_vl import Qwen3VLDense4BConfig
from xtuner.v1.train import Trainer

# Expected envs:
# - MODEL_PATH: HF checkpoint dir (also used as tokenizer_path/processor_path)
# - MEDIA_ROOT: media root for relative image/video paths (optional)
# - DATA_PATH_TEXT / DATA_PATH_VIDEO: jsonl file or directory (at least one must be non-empty)
model_path = os.environ["MODEL_PATH"]
media_root = os.environ.get("MEDIA_ROOT", "")
data_path_text = os.environ.get("DATA_PATH_TEXT", "")
data_path_video = os.environ.get("DATA_PATH_VIDEO", "")
assert data_path_text or data_path_video, "Please set DATA_PATH_TEXT and/or DATA_PATH_VIDEO"

model_cfg = Qwen3VLDense4BConfig()

sample_max_length = 8192
pack_max_length = 16384

dataset_config = []
if data_path_text:
    dataset_config.append(
        {
            "dataset": DatasetConfig(
                name="text",
                anno_path=data_path_text,
                sample_ratio=1.0,
                class_name="VLMJsonlDataset",
                media_root=media_root,
            ),
            "tokenize_fn": Qwen3VLTokenizeFnConfig(processor_path=model_path),
        }
    )
if data_path_video:
    dataset_config.append(
        {
            "dataset": DatasetConfig(
                name="video",
                anno_path=data_path_video,
                sample_ratio=1.0,
                class_name="VLMJsonlDataset",
                media_root=media_root,
            ),
            "tokenize_fn": Qwen3VLTokenizeFnConfig(processor_path=model_path),
        }
    )

dataloader_cfg = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_max_length=pack_max_length,
    num_workers=8,
    collator="qwen3_vl_sft_collator",
    enable_dataset_loss=True,
)

optim_cfg = AdamWConfig(lr=1e-6, foreach=False)
lr_cfg = LRConfig(lr_type="cosine", warmup_ratio=0)

trainer = Trainer(
    load_from=model_path,
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    dataloader_cfg=dataloader_cfg,
    lr_cfg=lr_cfg,
    tokenizer_path=model_path,
    fsdp_cfg=FSDPConfig(sp_size=1, recompute_ratio=1.0, torch_compile=True),
    global_batch_size=8,
    total_epoch=1,
    loss_cfg=CELossConfig(mode="chunk", chunk_size=1024),
    enable_dataset_loss=True,
)
trainer.fit()

