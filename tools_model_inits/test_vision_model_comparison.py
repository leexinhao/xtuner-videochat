#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：比较VideoChat3Vision Model和MoonViT模型的输出结果
"""
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
import sys
import os
import torch
import numpy as np
from PIL import Image
import json
from pathlib import Path
from transformers import activations
activations.PytorchGELUTanh = activations.GELUTanh


VideoChat_Path = "./VideoChat3-4B_t1"
MoonViT_Path = "/mnt/petrelfs/zengxiangyu/Research_lixinhao/models/MoonViT-SO-400M"
# 添加VideoChat3-debug目录到Python路径
sys.path.insert(0, VideoChat_Path)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, MoonViT_Path)

def load_moonvit_model():
    """加载MoonViT模型"""
    print("正在加载MoonViT模型...")
    
    model_path = MoonViT_Path
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    ).float()

    model.eval()
    config = model.config
    print(f"MoonViT模型加载完成，配置: {config}")
    return model, config

def load_videochat3_vision_model():
    """加载VideoChat3Vision模型"""
    print("正在加载VideoChat3Vision模型...")
    
    model_path = VideoChat_Path
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    ).vision_tower.float()

    model.eval()
    config = model.config

    print(f"VideoChat3Vision模型加载完成，配置: {config}")
    return model, config

def preprocess_image(image_path):
    """预处理图像"""
    print(f"正在处理图像: {image_path}")
    
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    
    # 调整图像大小到适合patch embedding的尺寸
    # 使用14x14的patch size，调整到合适的尺寸
    target_size = (448, 448)  # 32x32 patches
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # 转换为tensor
    image_array = np.array(image)
    image_tensor = torch.from_numpy(image_array).float()
    
    # 归一化到[0, 1]
    image_tensor = image_tensor / 255.0
    
    # 转换为CHW格式
    image_tensor = image_tensor.permute(2, 0, 1)
    
    # 添加batch维度
    image_tensor = image_tensor.unsqueeze(0)
    
    print(f"图像预处理完成，形状: {image_tensor.shape}")
    return image_tensor

def create_grid_thws(height, width, temporal=1):
    """创建grid_thws张量"""
    # 计算patch数量
    patch_size = 14
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    
    # 创建grid_thws张量 [temporal, height, width]
    grid_thws = torch.tensor([[temporal, num_patches_h, num_patches_w]], dtype=torch.long)
    
    print(f"Grid THW: {grid_thws}")
    return grid_thws

def compare_model_outputs():
    """比较两个模型的输出"""
    print("=" * 60)
    print("开始比较VideoChat3Vision和MoonViT模型输出")
    print("=" * 60)


    model_path = "./VideoChat3-4B_t1"
    processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)

    image_path = "tools_model_inits/mouth.png"
    image = Image.open(image_path)
    images_processed = processor(image, return_tensors="pt")
    # 数据放到cuda上
    images_processed = images_processed.to("cuda")
    print(images_processed.pixel_values.shape)

    image_tensor_moonvit = images_processed.pixel_values.reshape(-1, 3, 14, 14)
    grid_thws_moonvit = images_processed.image_grid_thw[:,1:]
    print(grid_thws_moonvit)

    print("\n" + "=" * 40)
    print("运行模型推理...")
    print("=" * 40)
    

    # 加载模型
    videochat3_model, videochat3_config = load_videochat3_vision_model()
    moonvit_model, moonvit_config = load_moonvit_model()

    with torch.no_grad():
        # MoonViT推理
        print("运行MoonViT模型...")
        moonvit_output = moonvit_model(image_tensor_moonvit, grid_thws_moonvit)
        print(f"MoonViT输出形状: {[out.shape for out in moonvit_output]}")
        print(f"MoonViT输出类型: {type(moonvit_output)}")
        
        # VideoChat3Vision推理
        print("运行VideoChat3Vision模型...")
        videochat3_output = videochat3_model(images_processed.pixel_values, images_processed.image_grid_thw)
        print(f"VideoChat3Vision输出形状: {[out.shape for out in videochat3_output]}")
        print(f"VideoChat3Vision输出类型: {type(videochat3_output)}")
    
    print("\n" + "=" * 40)
    print("比较结果...")
    print("=" * 40)
    
    # 比较输出
    if len(moonvit_output) == len(videochat3_output):
        print(f"两个模型输出数量相同: {len(moonvit_output)}")
        
        for i, (moonvit_out, videochat3_out) in enumerate(zip(moonvit_output, videochat3_output)):
            print(f"\n输出 {i}:")
            print(f"  MoonViT形状: {moonvit_out.shape}")
            print(f"  VideoChat3Vision形状: {videochat3_out.shape}")
            
            if moonvit_out.shape == videochat3_out.shape:
                # 计算差异
                diff = torch.abs(moonvit_out - videochat3_out)
                max_diff = torch.max(diff).item()
                mean_diff = torch.mean(diff).item()
                
                print(f"  最大差异: {max_diff:.6f}")
                print(f"  平均差异: {mean_diff:.6f}")
                
                # 检查是否接近
                tolerance = 1e-5
                if max_diff < tolerance:
                    print(f"  ✅ 输出 {i} 完全一致 (差异 < {tolerance})")
                else:
                    print(f"  ⚠️  输出 {i} 存在差异 (差异 >= {tolerance})")
                    
                    # 显示一些统计信息
                    print(f"  MoonViT输出范围: [{torch.min(moonvit_out).item():.6f}, {torch.max(moonvit_out).item():.6f}]")
                    print(f"  VideoChat3Vision输出范围: [{torch.min(videochat3_out).item():.6f}, {torch.max(videochat3_out).item():.6f}]")
            else:
                print(f"  ❌ 输出 {i} 形状不匹配")
    else:
        print(f"❌ 输出数量不匹配: MoonViT={len(moonvit_output)}, VideoChat3Vision={len(videochat3_output)}")
    
    print("\n" + "=" * 60)
    print("比较完成")
    print("=" * 60)

if __name__ == "__main__":
    try:
        compare_model_outputs()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
