#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多线程安全性测试脚本
验证多线程不会导致数据重复或丢失
"""

import os
import json
import pandas as pd
import argparse
from openai_api_tester import OpenAIAPITester


def test_thread_safety(api_key, dataset_path, image_folder, max_workers=10, test_count=20):
    """测试多线程安全性"""
    
    output_dir = "thread_safety_test"
    
    # 清理之前的测试
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    
    print(f"开始多线程安全性测试...")
    print(f"线程数: {max_workers}")
    print(f"测试样本数: {test_count}")
    print(f"输出目录: {output_dir}")
    
    # 创建测试器
    tester = OpenAIAPITester(
        api_key=api_key,
        base_url="https://c-z0-api-01.hash070.com/v1",
        model="gpt-4o",
        dataset_path=dataset_path,
        image_folder=image_folder,
        output_dir=output_dir,
        test_mode=True,
        test_count=test_count,
        multiprocess=True,
        max_workers=max_workers,
        resume=True  # 启用续传测试
    )
    
    print("\n第一次运行 (完整测试)...")
    results1 = tester.run()
    
    print(f"\n第一次运行完成，处理了 {len(results1)} 个样本")
    
    # 检查结果文件
    results_file = os.path.join(output_dir, "openai_gpt_4o_full_results.csv")
    progress_file = os.path.join(output_dir, "openai_gpt_4o_progress.json")
    
    if os.path.exists(results_file):
        df_results = pd.read_csv(results_file)
        print(f"CSV文件包含 {len(df_results)} 行数据")
        
        # 检查是否有重复的sample_id
        duplicate_ids = df_results['sample_id'].duplicated()
        if duplicate_ids.any():
            print(f"❌ 发现 {duplicate_ids.sum()} 个重复的sample_id:")
            print(df_results[duplicate_ids]['sample_id'].tolist())
        else:
            print("✅ 没有发现重复的sample_id")
    
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
            completed_ids = set(progress.get('completed_ids', []))
            print(f"进度文件记录了 {len(completed_ids)} 个已完成的ID")
    
    print("\n第二次运行 (测试续传)...")
    # 第二次运行应该跳过所有样本
    results2 = tester.run()
    
    print(f"第二次运行完成，处理了 {len(results2)} 个新样本 (应该为0)")
    
    # 再次检查结果文件
    if os.path.exists(results_file):
        df_results_after = pd.read_csv(results_file)
        print(f"第二次运行后CSV文件包含 {len(df_results_after)} 行数据")
        
        # 检查是否有新的重复
        duplicate_ids_after = df_results_after['sample_id'].duplicated()
        if duplicate_ids_after.any():
            print(f"❌ 第二次运行后发现 {duplicate_ids_after.sum()} 个重复的sample_id")
        else:
            print("✅ 第二次运行后仍然没有重复的sample_id")
        
        # 检查数据是否增加了
        if len(df_results_after) > len(df_results):
            print(f"❌ 续传失败：数据从 {len(df_results)} 增加到了 {len(df_results_after)}")
        else:
            print("✅ 续传成功：没有重复处理数据")
    
    # 现在测试部分中断的情况
    print("\n第三次运行 (模拟中断续传)...")
    
    # 修改进度文件，模拟只完成了一半
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        
        original_completed = progress['completed_ids']
        # 只保留一半的完成记录
        progress['completed_ids'] = original_completed[:len(original_completed)//2]
        
        with open(progress_file, 'w') as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)
        
        print(f"模拟中断：进度文件中只保留 {len(progress['completed_ids'])} 个已完成的ID")
    
    # 第三次运行
    results3 = tester.run()
    print(f"第三次运行完成，处理了 {len(results3)} 个新样本")
    
    # 最终检查
    if os.path.exists(results_file):
        df_final = pd.read_csv(results_file)
        print(f"最终CSV文件包含 {len(df_final)} 行数据")
        
        final_duplicates = df_final['sample_id'].duplicated()
        if final_duplicates.any():
            print(f"❌ 最终检查发现 {final_duplicates.sum()} 个重复的sample_id")
            return False
        else:
            print("✅ 最终检查：没有重复的sample_id")
            
        # 检查是否所有预期的样本都被处理了
        unique_samples = df_final['sample_id'].nunique()
        if unique_samples == test_count:
            print(f"✅ 所有 {test_count} 个样本都被正确处理")
            return True
        else:
            print(f"❌ 预期 {test_count} 个样本，但只有 {unique_samples} 个唯一样本")
            return False
    
    return False


def main():
    parser = argparse.ArgumentParser(description='多线程安全性测试')
    
    parser.add_argument('--api-key', required=True, help='API密钥')
    parser.add_argument('--dataset', required=True, help='数据集CSV文件路径')
    parser.add_argument('--image-folder', required=True, help='图片文件夹路径')
    parser.add_argument('--max-workers', type=int, default=10, help='最大线程数')
    parser.add_argument('--test-count', type=int, default=20, help='测试样本数')
    
    args = parser.parse_args()
    
    print("="*80)
    print("多线程安全性测试")
    print("="*80)
    print("该测试将验证:")
    print("1. 多线程不会导致样本重复处理")
    print("2. 续传功能正常工作") 
    print("3. 数据不会丢失或损坏")
    print("4. CSV文件写入是线程安全的")
    print("="*80)
    
    input("按Enter开始测试，或Ctrl+C取消...")
    
    success = test_thread_safety(
        api_key=args.api_key,
        dataset_path=args.dataset,
        image_folder=args.image_folder,
        max_workers=args.max_workers,
        test_count=args.test_count
    )
    
    print("\n" + "="*80)
    if success:
        print("🎉 多线程安全性测试通过！")
        print("可以安全地使用多线程模式。")
    else:
        print("❌ 多线程安全性测试失败！")
        print("建议使用单线程模式或检查代码。")
    print("="*80)


if __name__ == '__main__':
    main()
