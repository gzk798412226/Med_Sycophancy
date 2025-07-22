#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试脚本 - 验证环境和配置
"""

import os
import pandas as pd

def test_environment():
    """测试环境配置"""
    print("🔍 环境配置检查")
    print("=" * 50)
    
    # 1. 检查数据集
    dataset_path = "/home/zikun/github/Sycophency/NC_Scyphancy/NC_Challenge_benchmark/sycophancy_challenge_set.csv"
    print(f"📊 数据集检查: {dataset_path}")
    
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        print(f"  ✅ 数据集存在: {len(df)} 条记录")
        
        # 显示前3条记录的ID
        sample_ids = df['id'].head(3).tolist()
        print(f"  📝 前3个样本ID: {sample_ids}")
    else:
        print(f"  ❌ 数据集不存在")
        return False
    
    # 2. 检查图片文件夹
    image_folder = "/home/zikun/github/Sycophency/Medical_Sycophency/3datasets/merged_images"
    print(f"\n🖼️  图片文件夹检查: {image_folder}")
    
    if os.path.exists(image_folder):
        jpg_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
        print(f"  ✅ 图片文件夹存在: {len(jpg_files)} 张图片")
        
        # 检查前3个样本的图片是否存在
        for sample_id in sample_ids:
            image_path = os.path.join(image_folder, f"{sample_id}.jpg")
            if os.path.exists(image_path):
                print(f"  ✅ {sample_id}.jpg 存在")
            else:
                print(f"  ❌ {sample_id}.jpg 不存在")
    else:
        print(f"  ❌ 图片文件夹不存在")
        return False
    
    # 3. 检查必需的Python模块
    print(f"\n📦 Python依赖检查:")
    modules = ['openai', 'pandas']
    
    for module in modules:
        try:
            __import__(module)
            print(f"  ✅ {module} 已安装")
        except ImportError:
            print(f"  ❌ {module} 未安装")
            return False
    
    # 4. 检查模板文件
    print(f"\n📄 模板文件检查:")
    template_files = ['prompt_templates.py', 'create_multiple_choice.py']
    
    for file in template_files:
        if os.path.exists(file):
            print(f"  ✅ {file} 存在")
        else:
            print(f"  ❌ {file} 不存在")
            return False
    
    print(f"\n🎉 所有检查通过！环境配置正确。")
    return True

def run_quick_test():
    """运行快速API测试"""
    print(f"\n🚀 准备运行快速API测试")
    print("=" * 50)
    
    try:
        from openai_api_tester import OpenAIAPITester
        
        # 使用测试配置
        tester = OpenAIAPITester(
            api_key="sk-m6RCv7RNcBb5379c5412T3BlbKFJA6AC4d65Aace4AA5939e",
            base_url="https://aigptx.top/v1",
            model="gpt-4o",
            dataset_path="/home/zikun/github/Sycophency/NC_Scyphancy/NC_Challenge_benchmark/sycophancy_challenge_set.csv",
            image_folder="/home/zikun/github/Sycophency/Medical_Sycophency/3datasets/merged_images",
            output_dir="quick_test_results",
            test_mode=True,
            test_count=1,
            multiprocess=False,
            resume=False
        )
        
        print("⏳ 正在运行测试...")
        results = tester.run()
        
        if results:
            print(f"\n✅ 测试成功完成！")
            print(f"📊 处理了 {len(results)} 个样本")
            print(f"💰 Token消耗: {tester.total_tokens_used}")
            print(f"📞 API调用次数: {tester.api_calls_count}")
            print(f"\n🎯 现在可以运行完整测试了！")
        else:
            print(f"\n⚠️ 测试完成但没有结果")
            
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        return False
    
    return True

def show_usage_examples():
    """显示使用示例"""
    print(f"\n📚 使用示例")
    print("=" * 50)
    
    print("1️⃣ 测试模式（推荐先运行）:")
    print("python openai_api_tester.py \\")
    print("  --api-key 'sk-m6RCv7RNcBb5379c5412T3BlbKFJA6AC4d65Aace4AA5939e' \\")
    print("  --dataset 'sycophancy_challenge_set.csv' \\")
    print("  --test-mode \\")
    print("  --test-count 5")
    
    print(f"\n2️⃣ 完整测试（多进程）:")
    print("python openai_api_tester.py \\")
    print("  --api-key 'sk-m6RCv7RNcBb5379c5412T3BlbKFJA6AC4d65Aace4AA5939e' \\")
    print("  --dataset 'sycophancy_challenge_set.csv' \\")
    print("  --multiprocess \\")
    print("  --max-workers 3")
    
    print(f"\n3️⃣ Python脚本方式:")
    print("""
from openai_api_tester import OpenAIAPITester

tester = OpenAIAPITester(
    api_key="your-api-key",
    dataset_path="sycophancy_challenge_set.csv",
    test_mode=True,
    test_count=5
)

results = tester.run()
""")

def main():
    print("🔧 OpenAI API测试工具 - 环境验证")
    print("=" * 60)
    
    # 检查环境
    if test_environment():
        print(f"\n" + "="*60)
        
        # 询问是否运行快速测试
        response = input("🤔 是否运行快速API测试（1条数据验证API连接）？[y/N]: ").lower().strip()
        
        if response in ['y', 'yes', '是']:
            success = run_quick_test()
            if success:
                show_usage_examples()
        else:
            print(f"\n✅ 环境检查完成！")
            show_usage_examples()
    else:
        print(f"\n❌ 环境检查失败，请先解决上述问题。")

if __name__ == '__main__':
    main()
