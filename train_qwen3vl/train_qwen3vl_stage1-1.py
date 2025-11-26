from xtuner.v1.model import Qwen3VLDense4BConfig
from xtuner.v1.datasets import (
    DataloaderConfig,
    DatasetConfig,
    Qwen3VLTokenizeFnConfig
)
from xtuner.v1.config import LRConfig, AdamWConfig
from xtuner.v1.train import Trainer
from xtuner.v1.loss import CELossConfig
from xtuner.v1.config import LRConfig, AdamWConfig, FSDPConfig
from xtuner.v1.datasets.mllm_tokenize_fn import OSSLoaderConfig

model_cfg = Qwen3VLDense4BConfig()

oss_loader_cfg = OSSLoaderConfig(backend_kwargs={"conf_path": "~/petreloss.conf"})

sample_max_length = 8192 # 单条样本的最大长度，超过会被截断，并且会有警告输出
pack_max_length = 16384 # 训练一次 iter 所能包含的最大长度，pack 机制会尽可能将多条样本拼接在一起，减少 padding
# 如果你的显存不够，可以适当调小上述两个参数，但是请确保 sample_max_length <= pack_max_length

processor_path = "/mnt/petrelfs/zhuyuhan/workspace/checkpoints/Qwen3-VL-4B-Instruct"

dataset_config = [
    {
        "dataset": DatasetConfig(name='llava585k',
                                 anno_path='pnorm2:videochat3/videochat3_data_annoations/image/blip_laion_cc_sbu_558k.jsonl',
                                 media_root='pnorm2:s3://videochat3/image/LLaVA-Pretrain/',
                                 sample_ratio=1.0,
                                 class_name='VLMJsonlDataset'),
        "tokenize_fn": Qwen3VLTokenizeFnConfig(processor_path=processor_path,
                                                oss_loader_cfg=oss_loader_cfg),
    },
    {
        "dataset": DatasetConfig(name='Smit500k',
                                 anno_path='pnorm2:videochat3/videochat3_data_annoations/video/caption_smit_481k.jsonl',
                                 media_root='pvideo:s3://S-MiT/',
                                 sample_ratio=1.0,
                                 class_name='VLMJsonlDataset'),
        "tokenize_fn": Qwen3VLTokenizeFnConfig(processor_path=processor_path,
                                                oss_loader_cfg=oss_loader_cfg),
    },
    # {
    #     "dataset": DatasetConfig(name='video', # 数据别名
    #                              anno_path='/mnt/petrelfs/zhuyuhan/workspace/xtuner/tests/resource/mllm_sft_video_example_data.jsonl', # 多模态数据
    #                              media_root='/mnt/petrelfs/zhuyuhan/workspace/xtuner/tests/',
    #                              sample_ratio=10.0,
    #                              class_name='VLMJsonlDataset'),
    #     "tokenize_fn": Qwen3VLTokenizeFnConfig(
    #         processor_path=processor_path,
    #         video_max_pixels=512*2*4*16*16
    #     ),
    # }
]
# dataloader 配置
dataloader_config = DataloaderConfig(dataset_config_list=dataset_config,
                                     pack_max_length=pack_max_length, 
                                     num_workers=8,
                                     collator='qwen3_vl_sft_collator')

optim_cfg = AdamWConfig(lr=1e-6, foreach=False) # 不同模块的 device mesh 有差别，foreach 必须是 False
lr_cfg = LRConfig(lr_type="cosine", warmup_ratio=0)

load_from = "/mnt/petrelfs/zhuyuhan/workspace/checkpoints/Qwen3-VL-4B-Instruct" # 如果是微调模式，必须指定，否则会重头训练
tokenizer = "/mnt/petrelfs/zhuyuhan/workspace/checkpoints/Qwen3-VL-4B-Instruct"

trainer = Trainer(
    load_from=load_from, # 如果是微调模式，必须指定，否则会重头训练
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    tokenizer_path=tokenizer,
    fsdp_cfg = FSDPConfig(sp_size=1, recompute_ratio=1.0, torch_compile=False),
    # 全局 batch size
    # 假设是 8 卡训练，那么每张卡的 forward shape 是 (1, pack_max_length)，梯度累加次数是 1
    # 假设是 4 卡训练，那么每张卡的 forward shape 是 (1, pack_max_length)，梯度累加次数是 2 (自动折算)
    global_batch_size=8, 
    total_epoch=2,
    loss_cfg=CELossConfig(mode="chunk", chunk_size=1024), # 可以显著减少显存占用，推荐总是开启
)
trainer.fit()