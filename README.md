## VideoChat3 TBD

### Tokenize

|      Tokenize        | 文字 | 图片 | 多图 | 视频 | 多视频 |
|-------------------|:----:|:----:|:----:|:----:|:------:|
| Qwen3VL    |  ✅  |  ✅  |   ✅  |   ✅   |    ✅     |
| VideoChat3    |   ✅   |   ✅   |   ✅   |   ✅   |   ✅   |

### Model 

|      Model        | 文字 | 图片 | 多图 | 视频 | 多视频 |
|-------------------|:----:|:----:|:----:|:----:|:------:|
| Qwen3VL-Dense     |  ✅  |  ✅  |  🚧  |  🚧  |   🚧   |
| Qwen3VL-MoE       |  ✅  |  ✅  |  🚧  |  🚧  |   🚧   |
| VideoChat3-Dense  |  ✅  | ✅🚧 (no fsdp) | 🚧 | ✅🚧 (no fsdp) | 🚧 |
| VideoChat3-MoE    |  🚧  |  🚧  |  🚧  |  🚧  |   🚧   |







## 📖 XTuner V1

XTuner V1 is a next-generation LLM training engine specifically designed for ultra-large-scale MoE models. Unlike traditional 3D parallel training architectures, XTuner V1 is optimized for the mainstream MoE training scenarios prevalent in today's academic research.

### Key Features
**📊 Dropless Training**
	
  - **Scalable without complexity:** Train 200B-scale MoE models without expert parallelism; 600B models require only intra-node expert parallelism	
  - **Optimized parallelism strategy:** Smaller expert parallelism dimension compared to traditional 3D approaches, enabling more efficient Dropless training

**📝 Long Sequence Support**
	
  - **Memory-efficient design:** Train 200B MoE models on 64k sequence lengths without sequence parallelism through advanced memory optimization techniques	
  - **Flexible scaling:** Full support for DeepSpeed Ulysses sequence parallelism with linearly scalable maximum sequence length	
  - **Robust performance:** Maintains stability despite expert load imbalance during long sequence training

**⚡ Superior Efficiency**

  - **Massive scale:** Supports MoE training up to 1T parameters	
  - **Breakthrough performance:** First to achieve FSDP training throughput that surpasses traditional 3D parallel schemes for MoE models above 200B scale
  - **Hardware optimization:** Achieves training efficiency on Ascend A3 Supernode that exceeds NVIDIA H800


<div align=center>
  <img src="https://github.com/user-attachments/assets/c4fb2bb4-56bd-4f1c-8188-7f5370314cf8" style="width:90%">
</div>


## 🔥 Roadmap

XTuner V1 is committed to continuously improving training efficiency for pre-training, instruction fine-tuning, and reinforcement learning of ultra-large MoE models, with special focus on Ascend NPU optimization.

### 🚀 Training Engine

Our vision is to establish XTuner V1 as a versatile training backend that seamlessly integrates with the broader open-source ecosystem.


|   Model    |  GPU(FP8) | GPU(BF16)| NPU(BF16) |
|------------|-----------|----------|-----------|
| Intern S1  |    ✅     |    ✅    |    ✅     |
| Intern VL  |    ✅     |    ✅    |    ✅     |
| Qwen3 Dense|    ✅     |    ✅    |    ✅     |
| Qwen3 MoE  |    ✅     |    ✅    |    ✅     |
| GPT OSS    |    ✅     |    ✅    |    🚧     |
| Deepseek V3|    ✅     |    ✅    |    🚧     |
| KIMI K2    |    ✅     |    ✅    |    🚧     |


### 🧠 Algorithm

The algorithm component is actively evolving. We welcome community contributions - with XTuner V1, scale your algorithms to unprecedented sizes!

**Implemented**


- ✅ **Multimodal Pre-training** - Full support for vision-language model training
- ✅ **Multimodal Supervised Fine-tuning** - Optimized for instruction following	
- ✅ [GRPO](https://arxiv.org/pdf/2402.03300) - Group Relative Policy Optimization


**Coming Soon**

- 🔄 [MPO](https://arxiv.org/pdf/2411.10442) - Mixed Preference Optimization
- 🔄 [DAPO](https://arxiv.org/pdf/2503.14476) - Dynamic Sampling Policy Optimization
- 🔄 **Multi-turn Agentic RL** - Advanced agent training capabilities


### ⚡ Inference Engine Integration

Seamless deployment with leading inference frameworks:
- [x] LMDeploy
- [ ] vLLM
- [ ] SGLang



### Data Preparation

- You can use [GraphGen](https://github.com/open-sciencelab/GraphGen) to create synthetic data for fine-tuning.

## 🤝 Contributing

We appreciate all contributions to XTuner. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## 🙏 Acknowledgement

The development of XTuner V1's training engine has been greatly inspired by and built upon the excellent work of the open-source community. We extend our sincere gratitude to the following pioneering projects:

**Training Engine:**

- [Torchtitan](https://github.com/pytorch/torchtitan) - A PyTorch native platform for training generative AI models
- [Deepspeed](https://github.com/deepspeedai/DeepSpeed) - Microsoft's deep learning optimization library	
- [MindSpeed](https://gitee.com/ascend/MindSpeed) - Ascend's high-performance training acceleration library	
- [Megatron](https://github.com/NVIDIA/Megatron-LM) - NVIDIA's large-scale transformer training framework


**Reinforcement Learning:**

XTuner V1's reinforcement learning capabilities have been enhanced through insights and best practices from:

- [veRL](https://github.com/volcengine/verl) - Volcano Engine Reinforcement Learning for LLMs	
- [SLIME](https://github.com/THUDM/slime) - THU's scalable RLHF implementation	
- [AReal](https://github.com/inclusionAI/AReaL) - Ant Reasoning Reinforcement Learning for LLMs
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) - An Easy-to-use, Scalable and High-performance RLHF Framework based on Ray

We are deeply grateful to all contributors and maintainers of these projects for advancing the field of large-scale model training.


## 🖊️ Citation

```bibtex
@misc{2023xtuner,
    title={XTuner: A Toolkit for Efficiently Fine-tuning LLM},
    author={XTuner Contributors},
    howpublished = {\url{https://github.com/InternLM/xtuner}},
    year={2023}
}
```

## License

This project is released under the [Apache License 2.0](LICENSE). Please also adhere to the Licenses of models and datasets being used.
