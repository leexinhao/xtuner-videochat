#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VideoChat3模型权重初始化脚本

使用方法：
python init_model_weights.py

该脚本会：
1. 从ViT-SO-400M加载vision权重
2. 从Qwen3-4B-Instruct-2507加载language权重  
3. 创建完整的VideoChat3模型并保存
"""

import os
import sys
import json
import torch
from pathlib import Path
from transformers import AutoModel, AutoConfig
from safetensors import safe_open

# 导入VideoChat3相关模块
from configuration_videochat3 import VideoChat3Config
from modeling_videochat3 import VideoChat3ForConditionalGeneration


def load_weights_from_safetensors(model_path: str):
    """从safetensors文件加载权重"""
    weights = {}
    
    # 检查单个safetensors文件
    single_file = os.path.join(model_path, "model.safetensors")
    if os.path.exists(single_file):
        print(f"加载单个权重文件: {single_file}")
        with safe_open(single_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
        return weights
    
    # 检查多个safetensors文件
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        print(f"加载多个权重文件，索引: {index_file}")
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        
        for weight_file in index_data["weight_map"].values():
            file_path = os.path.join(model_path, weight_file)
            if os.path.exists(file_path):
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        weights[key] = f.get_tensor(key)
        return weights
    
    raise FileNotFoundError(f"未找到权重文件在: {model_path}")


def main():
    """主函数"""
    print("VideoChat3模型权重初始化")
    print("=" * 40)
    
    # 设置路径
    vit_path = "ViT-SO-400M"
    qwen3_path = "Qwen3-4B-Instruct-2507"
    current_dir = ""
    output_path = os.path.join(current_dir, "initialized_model")
    
    try:
        # 1. 检查输入路径
        print("检查输入路径...")
        if not os.path.exists(vit_path):
            raise FileNotFoundError(f"MoonViT路径不存在: {vit_path}")
        if not os.path.exists(qwen3_path):
            raise FileNotFoundError(f"Qwen3路径不存在: {qwen3_path}")
        print("✅ 输入路径检查通过")
        
        # 2. 加载配置
        print("\n加载VideoChat3配置...")
        config_path = os.path.join(current_dir, "config.json")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = VideoChat3Config.from_dict(config_dict)
        print("✅ 配置加载完成")
        
        # 3. 创建模型
        print("\n创建VideoChat3模型...")
        model = VideoChat3ForConditionalGeneration(config)
        print("✅ 模型创建完成")
        
        # 4. 加载预训练权重
        print("\n加载预训练权重...")
        
        # 加载MoonViT权重
        print("  加载MoonViT权重...")
        vit_weights = load_weights_from_safetensors(vit_path)
        print(f"  ✅ 加载了 {len(vit_weights)} 个MoonViT权重")
        
        # 加载Qwen3权重
        print("  加载Qwen3权重...")
        qwen3_weights = load_weights_from_safetensors(qwen3_path)
        print(f"  ✅ 加载了 {len(qwen3_weights)} 个Qwen3权重")
        
        # 5. 构建完整的状态字典
        print("\n构建模型状态字典...")
        state_dict = {}
        
        # 添加Vision权重
        for key, tensor in vit_weights.items():
            state_dict[f"model.vision_tower.{key}"] = tensor
        
        # 添加Language权重
        for key, tensor in qwen3_weights.items():
            state_dict[f"model.language_model.{key}"] = tensor
        
        print(f"✅ 状态字典构建完成，共 {len(state_dict)} 个权重")
        
        # 6. 加载权重到模型
        print("\n加载权重到模型...")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️  缺失 {len(missing_keys)} 个权重（这是正常的，因为有些权重需要默认初始化）")
        
        if unexpected_keys:
            print(f"⚠️  未使用 {len(unexpected_keys)} 个权重")
        
        print("✅ 权重加载完成")
        
        # 7. 测试模型
        print("\n测试模型...")
        model.eval()
        
        # 创建测试输入
        batch_size = 1
        seq_len = 5
        vocab_size = model.config.text_config.vocab_size
        
        input_ids = torch.randint(0, min(vocab_size, 1000), (batch_size, seq_len))
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            print(f"✅ 模型测试成功! 输出形状: {outputs.logits.shape}")
        
        # 8. 保存模型
        print(f"\n保存模型到: {output_path}")
        os.makedirs(output_path, exist_ok=True)
        model.save_pretrained(output_path)
        print("✅ 模型保存完成")
        
        # 9. 显示模型信息
        print("\n" + "=" * 40)
        print("🎉 VideoChat3模型初始化完成!")
        print(f"📁 模型保存位置: {output_path}")
        print(f"📊 模型配置:")
        print(f"   - Vision模型: {config.vision_config.model_type}")
        print(f"   - Text模型: {config.text_config.model_type}")
        print(f"   - 总参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
