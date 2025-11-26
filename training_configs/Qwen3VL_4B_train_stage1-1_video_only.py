from xtuner.v1.config import (
    AdamWConfig,
    LRConfig,
)
from xtuner.v1.train import TrainerConfig, ResumeConfig
from xtuner.v1.datasets import Qwen3VLTokenizeFnConfig
from xtuner.v1.model import Qwen3VLDense4BConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.datasets.config import DatasetConfig, DataloaderConfig
from xtuner.v1.datasets.mllm_tokenize_fn import OSSLoaderConfig
from xtuner.v1.config import FSDPConfig
import json

# export XTUNER_TOKENIZE_WORKERS=16
# export XTUNER_USE_FA3=1

# model config
model_cfg = Qwen3VLDense4BConfig(freeze_vision=True, freeze_projector=True)

model_path = "/mnt/petrelfs/zengxiangyu/Research_lixinhao/models/Qwen3-VL-4B-Instruct"
meta_data_path = '/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/data_stage1-1_video_only.json'
work_dir = "work_dir/Qwen3VL_4B_train_stage1-1_video_only"
cache_dir = "dataset_cache/cache_qwen3vl_4B_stage1-1_video_only"

sample_max_length = 8192
pack_max_length = 8192
global_batch_size = 128
total_epoch = 1
# total_num_tokens = 105332882
# total_step = int(total_num_tokens  / pack_max_length / global_batch_size)
hf_interval = 100
checkpoint_interval = 100
checkpoint_maxkeep = 2

lr = 1e-3
weight_decay = 0.0
warmup_ratio = 0.03
lr_min = 1e-6
recompute_ratio = 1.0
loss_reduction = "square"

oss_loader_cfg = OSSLoaderConfig(backend_kwargs={"conf_path": "~/petreloss.conf"})
# oss_loader_cfg=None
ds_collections = json.loads(open(meta_data_path).read())
dataset_config = []
for name, _data in ds_collections.items():
    _data_cfg = {
        "dataset": DatasetConfig(
            name=name,
            anno_path=_data['anno_path'],
            media_root=_data.get('media_root', ''),
            sample_ratio=_data.get('sample_ratio', 1.0),
            class_name='VLMJsonlDataset',
            cache_dir=cache_dir
        ),
        "tokenize_fn": Qwen3VLTokenizeFnConfig(
            max_length=sample_max_length,
            min_pixels=_data.get('image_min_pixels', 28*28),
            max_pixels=_data.get('image_max_pixels', int(sample_max_length * 0.8 * 28 * 28)),
            video_max_total_pixels=_data.get('video_max_total_pixels', int(sample_max_length * 0.8 * 4 * 28 * 28)),
            video_min_frames=_data.get('video_min_frames', 1),
            video_max_frames=_data.get('video_max_frames', 2048),
            fps=_data.get('video_sample_fps', 4),
            processor_path=model_path,
            system_message=_data.get('system_message', None),
            hash=_data.get('hash', None),
            oss_loader_cfg=oss_loader_cfg
        )
    }
    dataset_config.append(_data_cfg)

dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_max_length=pack_max_length,
    collator="qwen3_vl_sft_collator",
    num_workers=8,
    pack_extra_buffer_size=20,
)

# optimizer and lr config
optim_cfg = AdamWConfig(lr=lr, weight_decay=weight_decay, foreach=False)
lr_cfg = LRConfig(lr_type="cosine", warmup_ratio=warmup_ratio, lr_min=lr_min)
fsdp_cfg = FSDPConfig(sp_size=1, recompute_ratio=recompute_ratio, torch_compile=False)

resume_cfg = ResumeConfig(auto_resume=False)

# trainer config
trainer = TrainerConfig(
    load_from=model_path,
    resume_cfg=resume_cfg,
    tokenizer_path=model_path,
    fsdp_cfg=fsdp_cfg,
    exp_tracker='tensorboard',
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=CELossConfig(mode="chunk", chunk_size=1024, loss_reduction=loss_reduction),
    global_batch_size=global_batch_size,
    total_epoch=total_epoch,
    hf_interval=hf_interval,
    checkpoint_interval=checkpoint_interval,
    checkpoint_maxkeep=checkpoint_maxkeep,
    work_dir=work_dir,
)