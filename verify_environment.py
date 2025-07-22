#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境验证脚本
检查所有依赖和配置是否正确
"""

import sys
import os
import pandas as pd

def check_imports():
    """检查必要的包是否已安装"""
    print("检查Python包...")
    
    required_packages = [
        ('openai', 'openai'),
        ('pandas', 'pandas'),
        ('base64', 'base64'),
        ('json', 'json'),
        ('re', 're'),
        ('concurrent.futures', 'concurrent.futures'),
        ('threading', 'threading')
    ]
    
    missing_packages = []
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} - 未安装")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n❌ 缺少以下包: {', '.join(missing_packages)}")
        return False
    else:
        print("\n✅ 所有必要的包都已安装")
        return True

def check_local_files():
    """检查本地文件是否存在"""
    print("\n检查本地文件...")
    
    current_dir = "/home/zikun/github/Sycophency/NC_Scyphancy/NC_Challenge_benchmark"
    
    required_files = [
        'prompt_templates.py',
        'create_multiple_choice.py',
        'openai_api_tester.py',
        'sycophancy_challenge_set.csv'
    ]
    
    missing_files = []
    
    for filename in required_files:
        filepath = os.path.join(current_dir, filename)
        if os.path.exists(filepath):
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} - 文件不存在")
            missing_files.append(filename)
    
    if missing_files:
        print(f"\n❌ 缺少以下文件: {', '.join(missing_files)}")
        return False
    else:
        print("\n✅ 所有必要的文件都存在")
        return True

def check_dataset():
    """检查数据集格式"""
    print("\n检查数据集...")
    
    dataset_path = "/home/zikun/github/Sycophency/NC_Scyphancy/NC_Challenge_benchmark/sycophancy_challenge_set.csv"
    
    try:
        df = pd.read_csv(dataset_path)
        print(f"  ✓ 数据集加载成功")
        print(f"  ✓ 数据集大小: {len(df)} 行")
        
        required_columns = ['id', 'question', 'correct_answer']
        missing_columns = []
        
        for col in required_columns:
            if col in df.columns:
                print(f"  ✓ 列 '{col}' 存在")
            else:
                print(f"  ✗ 列 '{col}' 不存在")
                missing_columns.append(col)
        
        if missing_columns:
            print(f"\n❌ 数据集缺少必要的列: {', '.join(missing_columns)}")
            return False
        else:
            print("\n✅ 数据集格式正确")
            print(f"数据集预览:")
            print(df.head(3)[['id', 'question', 'correct_answer']].to_string(index=False))
            return True
            
    except Exception as e:
        print(f"  ✗ 数据集加载失败: {str(e)}")
        return False

def check_image_folder():
    """检查图片文件夹"""
    print("\n检查图片文件夹...")
    
    image_folder = "/home/zikun/github/Sycophency/NC_Scyphancy/NC_Challenge_benchmark/challenge_dataset"
    
    if os.path.exists(image_folder):
        image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  ✓ 图片文件夹存在: {image_folder}")
        print(f"  ✓ 发现 {len(image_files)} 个图片文件")
        
        if len(image_files) > 0:
            print(f"  示例图片: {image_files[:3]}")
            return True
        else:
            print(f"  ⚠️ 图片文件夹为空")
            return False
    else:
        print(f"  ✗ 图片文件夹不存在: {image_folder}")
        return False

def main():
    print("=== OpenAI API测试工具环境验证 ===")
    print(f"Python版本: {sys.version}")
    print(f"当前工作目录: {os.getcwd()}")
    
    checks = [
        check_imports(),
        check_local_files(), 
        check_dataset(),
        check_image_folder()
    ]
    
    print(f"\n=== 验证结果 ===")
    if all(checks):
        print("🎉 所有检查都通过！环境配置正确，可以开始使用API测试工具。")
        print("\n建议的下一步:")
        print("1. 检查API密钥是否正确")
        print("2. 运行测试模式: python openai_api_tester.py --api-key YOUR_KEY --dataset sycophancy_challenge_set.csv --test-mode")
        return True
    else:
        print("❌ 环境验证失败，请修复上述问题后重试。")
        return False

if __name__ == "__main__":
    main()
