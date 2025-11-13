from xtuner.v1.model import VideoChat3Dense2BConfig
from xtuner.v1.datasets import (
    DataloaderConfig,
    DatasetConfig,
    VideoChat3TokenizeFnConfig
)
from xtuner.v1.config import LRConfig, AdamWConfig, FSDPConfig
from xtuner.v1.train import Trainer
from xtuner.v1.loss import CELossConfig


model_cfg = VideoChat3Dense2BConfig()

sample_max_length = 8192 # 单条样本的最大长度，超过会被截断，并且会有警告输出
pack_max_length = 16384 # 训练一次 iter 所能包含的最大长度，pack 机制会尽可能将多条样本拼接在一起，减少 padding
# 如果你的显存不够，可以适当调小上述两个参数，但是请确保 sample_max_length <= pack_max_length

processor_path = "/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/VideoChat3-2B"

dataset_config = [
    {
        "dataset": DatasetConfig(name='pure_text', # 数据别名
                                 # 标注文件路径，可以是单个 jsonl 也可以是文件夹，会自动遍历当前文件夹下所有 jsonl 文件
                                 anno_path='/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/tests/resource/mllm_sft_text_example_data.jsonl', # 纯文本数据
                                 sample_ratio=10.0, # 数据采样比例，这里是重复 5 遍，可以是小数
                                 class_name='VLMJsonlDataset'), # 对应的 dataset 类名
        # 一个 dataset 要配一个对应的 tokenizer fun 函数用于处理 dataset 输出的单条 item 数据
        "tokenize_fn": VideoChat3TokenizeFnConfig(processor_path=processor_path),
    },
    {
        "dataset": DatasetConfig(name='media', # 数据别名
                                 anno_path='/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/tests/resource/mllm_sft_single_image_example_data.jsonl', # 多模态数据
                                 media_root='/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/tests/',
                                 sample_ratio=10.0,
                                 class_name='VLMJsonlDataset'),
        "tokenize_fn": VideoChat3TokenizeFnConfig(processor_path=processor_path),
    },
    {
        "dataset": DatasetConfig(name='video', # 数据别名
                                 anno_path='/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/tests/resource/mllm_sft_video_example_data.jsonl', # 多模态数据
                                 media_root='/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/tests/',
                                 sample_ratio=10.0,
                                 class_name='VLMJsonlDataset'),
        "tokenize_fn": VideoChat3TokenizeFnConfig(
            processor_path=processor_path,
            video_max_pixels=512*2*4*16*16
        ),
    }
]
# dataloader 配置
dataloader_config = DataloaderConfig(dataset_config_list=dataset_config,
                                     pack_max_length=pack_max_length, 
                                     num_workers=8,
                                     collator='videochat3_sft_collator')

optim_cfg = AdamWConfig(lr=1e-6, foreach=False) # 不同模块的 device mesh 有差别，foreach 必须是 False
lr_cfg = LRConfig(lr_type="cosine", warmup_ratio=0)


load_from = "/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/VideoChat3-2B" # 如果是微调模式，必须指定，否则会重头训练
tokenizer = "/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/VideoChat3-2B"

trainer = Trainer(
    load_from=load_from, # 如果是微调模式，必须指定，否则会重头训练
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    tokenizer_path=tokenizer,
    fsdp_cfg = FSDPConfig(sp_size=1, recompute_ratio=1.0, torch_compile=True),
    # 全局 batch size
    # 假设是 8 卡训练，那么每张卡的 forward shape 是 (1, pack_max_length)，梯度累加次数是 1
    # 假设是 4 卡训练，那么每张卡的 forward shape 是 (1, pack_max_length)，梯度累加次数是 2 (自动折算)
    global_batch_size=8, 
    total_epoch=2,
    loss_cfg=CELossConfig(mode="chunk", chunk_size=1024), # 可以显著减少显存占用，推荐总是开启
)
trainer.fit()