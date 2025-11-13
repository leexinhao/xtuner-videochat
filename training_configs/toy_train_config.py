from xtuner.v1.config import (
    AdamWConfig,
    LRConfig,
)
from xtuner.v1.train import TrainerConfig, ResumeConfig
from xtuner.v1.datasets import VideoChat3TokenizeFnConfig
from xtuner.v1.model import VideoChat3Dense2BConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.datasets.config import DatasetConfig, DataloaderConfig
from xtuner.v1.datasets.mllm_tokenize_fn import OSSLoaderConfig
from xtuner.v1.config import FSDPConfig
import json

# export XTUNER_TOKENIZE_WORKERS=16
# export XTUNER_USE_FA3=1

# model config
model_cfg = VideoChat3Dense2BConfig()

model_path = "/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/VideoChat3-2B"
meta_data_path = '/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/training_data_annotations/data_debug.json'
work_dir = "work_dir/videochat3_2B_deug"
cache_dir = "dataset_cache/cache_videochat3_2B_deug"

global_batch_size = 8
total_step = 8000
hf_interval = 1000
checkpoint_interval = 1000
checkpoint_maxkeep = 10
sample_max_length = 32768
pack_max_length = 32768
lr = 8e-5
weight_decay = 0.05
warmup_ratio = 0.1
lr_min = 1e-6
recompute_ratio = 1.0
loss_reduction = "square"

# oss_loader_cfg = OSSLoaderConfig(backend_kwargs={"conf_path": "/mnt/shared-storage-user/huanghaian/petreloss.conf"})
oss_loader_cfg=None
ds_collections = json.loads(open(meta_data_path).read())
dataset_config = []
for name, _data in ds_collections.items():
    _data_cfg = {"dataset": DatasetConfig(name=name,
                                          anno_path=_data['anno_path'],
                                          media_root=_data.get('media_root', ''),
                                          sample_ratio=_data.get('sample_ratio', 1.0),
                                          class_name='VLMJsonlDataset',
                                          cache_dir=cache_dir),
                 "tokenize_fn": VideoChat3TokenizeFnConfig(model_cfg=model_cfg,
                                                           max_length=sample_max_length,
                                                           processor_path=model_path,
                                                           video_max_pixels=512*2*4*16*16,
                                                           data_augment=_data.get('data_augment', False),
                                                           system_message=_data.get('system_message', None),
                                                           hash=_data.get('hash', None),
                                                           oss_loader_cfg=oss_loader_cfg
                                                           )
                 }
    dataset_config.append(_data_cfg)

dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_max_length=pack_max_length,
    collator="videochat3_sft_collator",
    num_workers=8,
    pack_extra_buffer_size=20,
)

# optimizer and lr config
optim_cfg = AdamWConfig(lr=lr, weight_decay=weight_decay, foreach=False)
lr_cfg = LRConfig(lr_type="cosine", warmup_ratio=warmup_ratio, lr_min=lr_min)
fsdp_cfg = FSDPConfig(sp_size=1, recompute_ratio=recompute_ratio, torch_compile=False)

resume_cfg = ResumeConfig(auto_resume=True)

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
    total_step=total_step,
    hf_interval=hf_interval,
    checkpoint_interval=checkpoint_interval,
    checkpoint_maxkeep=checkpoint_maxkeep,
    work_dir=work_dir,
)