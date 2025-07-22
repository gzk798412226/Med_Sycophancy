#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速验证和测试脚本
用于验证配置和运行小规模测试
"""

import os
import pandas as pd

def check_environment():
    """检查环境配置"""
    print("环境检查:")
    
    # 检查数据集文件
    dataset_path = "/home/zikun/github/Sycophency/NC_Scyphancy/NC_Challenge_benchmark/sycophancy_challenge_set.csv"
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        print(f"✅ 数据集文件存在: {len(df)} 条数据")
        print(f"   路径: {dataset_path}")
        
        # 检查前几条数据
        print(f"   前3条样本ID: {list(df['id'].head(3))}")
    else:
        print(f"❌ 数据集文件不存在: {dataset_path}")
        return False
    
    # 检查图片文件夹
    image_folder = "challenge_dataset"
    if os.path.exists(image_folder):
        image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
        print(f"✅ 图片文件夹存在: {len(image_files)} 张图片")
        print(f"   路径: {os.path.abspath(image_folder)}")
    else:
        print(f"❌ 图片文件夹不存在: {image_folder}")
        return False
    
    # 检查必需的Python模块
    try:
        import openai
        print("✅ openai 模块已安装")
    except ImportError:
        print("❌ openai 模块未安装，请运行: pip install openai")
        return False
    
    try:
        import pandas
        print("✅ pandas 模块已安装")
    except ImportError:
        print("❌ pandas 模块未安装，请运行: pip install pandas")
        return False
    
    # 检查模板文件
    required_files = ['prompt_templates.py', 'create_multiple_choice.py']
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} 存在")
        else:
            print(f"❌ {file} 不存在")
            return False
    
    return True

def run_quick_test():
    """运行快速测试"""
    if not check_environment():
        print("\n环境检查失败，请先解决上述问题")
        return
    
    print(f"\n{'='*60}")
    print("运行快速测试（测试模式，1条数据）")
    print(f"{'='*60}")
    
    # 导入测试器
    try:
        from openai_api_tester import OpenAIAPITester
    except ImportError as e:
        print(f"❌ 导入测试器失败: {e}")
        return
    
    # 配置
    API_KEY = "sk-m6RCv7RNcBb5379c5412T3BlbKFJA6AC4d65Aace4AA5939e"
    DATASET_PATH = "/home/zikun/github/Sycophency/NC_Scyphancy/NC_Challenge_benchmark/sycophancy_challenge_set.csv"
    
    # 创建测试器
    tester = OpenAIAPITester(
        api_key=API_KEY,
        base_url="https://aigptx.top/v1",
        model="gpt-4o",
        dataset_path=DATASET_PATH,
        image_folder="challenge_dataset",
        output_dir="quick_test_results",
        test_mode=True,
        test_count=1,  # 只测试1条数据
        multiprocess=False,
        resume=False   # 不使用断点续传
    )
    
    print("开始测试...")
    try:
        results = tester.run()
        print("\n✅ 快速测试完成！")
        print("现在可以运行完整的测试了")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        print("请检查API密钥是否正确，网络是否正常")

def main():
    print("OpenAI API测试工具 - 快速验证脚本")
    print("=" * 60)
    
    # 首先检查环境
    if check_environment():
        print(f"\n✅ 环境检查通过！")
        
        # 询问是否运行快速测试
        response = input("\n是否运行快速测试（1条数据）？(y/n): ").lower()
        if response == 'y' or response == 'yes':
            run_quick_test()
        else:
            print("\n准备就绪！可以使用以下命令运行测试：")
            print("\n# 测试模式（5条数据）：")
            print("python openai_api_tester.py --api-key 'your-key' --dataset 'sycophancy_challenge_set.csv' --test-mode")
            print("\n# 完整测试（多进程）：")
            print("python openai_api_tester.py --api-key 'your-key' --dataset 'sycophancy_challenge_set.csv' --multiprocess")

if __name__ == '__main__':
    main()
