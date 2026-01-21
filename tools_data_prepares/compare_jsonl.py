import json
import sys
from collections import deque
from pathlib import Path

def get_last_n_lines(file_path, n=10):
    """读取文件的最后n行"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 使用deque高效获取最后n行
            last_lines = deque(f, maxlen=n)
            return list(last_lines)
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 不存在")
        return []
    except Exception as e:
        print(f"读取文件 '{file_path}' 时出错: {e}")
        return []

def compare_jsonl_files(file1, file2, n=10):
    """比较两个JSONL文件的最后n行"""
    print(f"比较文件:")
    print(f"1. {file1}")
    print(f"2. {file2}")
    print(f"比较最后 {n} 行\n")
    print("="*60)
    
    # 获取最后n行
    lines1 = get_last_n_lines(file1, n)
    lines2 = get_last_n_lines(file2, n)
    
    if not lines1 or not lines2:
        return
    
    # 显示每个文件的实际行数
    print(f"文件1最后{len(lines1)}行，文件2最后{len(lines2)}行")
    
    # 比较行数
    min_len = min(len(lines1), len(lines2))
    
    # 逐行比较
    differences = []
    for i in range(min_len):
        line_num = i + 1
        line1 = lines1[i].strip()
        line2 = lines2[i].strip()
        
        try:
            # 尝试解析JSON
            json1 = json.loads(line1) if line1 else None
            json2 = json.loads(line2) if line2 else None
            
            if json1 != json2:
                differences.append({
                    'line': line_num,
                    'file1': line1,
                    'file2': line2,
                    'json1': json1,
                    'json2': json2
                })
        except json.JSONDecodeError:
            # 如果不是有效的JSON，直接比较字符串
            if line1 != line2:
                differences.append({
                    'line': line_num,
                    'file1': line1,
                    'file2': line2,
                    'json1': None,
                    'json2': None
                })
    
    # 如果两个文件的行数不同
    if len(lines1) != len(lines2):
        print(f"\n⚠️  文件行数不同:")
        print(f"文件1有{len(lines1)}行，文件2有{len(lines2)}行")
        
        if len(lines1) > len(lines2):
            print(f"文件1独有的行:")
            for i in range(len(lines2), len(lines1)):
                print(f"  第{i+1}行: {lines1[i].strip()}")
        else:
            print(f"文件2独有的行:")
            for i in range(len(lines1), len(lines2)):
                print(f"  第{i+1}行: {lines2[i].strip()}")
    
    # 显示差异
    if differences:
        print(f"\n❌ 发现 {len(differences)} 处差异:")
        for diff in differences:
            print(f"\n--- 第 {diff['line']} 行 ---")
            print(f"文件1: {diff['file1']}")
            print(f"文件2: {diff['file2']}")
            
            # 如果是JSON，显示结构化差异
            if diff['json1'] is not None and diff['json2'] is not None:
                print("JSON差异分析:")
                keys1 = set(diff['json1'].keys()) if isinstance(diff['json1'], dict) else set()
                keys2 = set(diff['json2'].keys()) if isinstance(diff['json2'], dict) else set()
                
                # 检查键的差异
                if keys1 != keys2:
                    print(f"  键不同 - 文件1有: {keys1 - keys2}，文件2有: {keys2 - keys1}")
                
                # 检查值的差异
                common_keys = keys1 & keys2
                for key in common_keys:
                    if diff['json1'].get(key) != diff['json2'].get(key):
                        print(f"  键 '{key}' 的值不同:")
                        print(f"    文件1: {diff['json1'].get(key)}")
                        print(f"    文件2: {diff['json2'].get(key)}")
    else:
        if len(lines1) == len(lines2):
            print("\n✅ 两个文件的最后10行完全相同！")
        else:
            print(f"\n✅ 两个文件的最后{min_len}行相同，但行数不同")
    
    print("="*60)
    print("比较完成")

file1_path = "/mnt/petrelfs/zengxiangyu/Research_lixinhao/videochat3_data_annoations/image/cc3m_xtuner_qwen3_recap_clean_repeat.jsonl"
file2_path = "/mnt/petrelfs/zengxiangyu/Research_lixinhao/videochat3_data_annoations/image/cc3m_xtuner_qwen3_recap_clean_repeat_new.jsonl"
compare_jsonl_files(file1_path, file2_path)