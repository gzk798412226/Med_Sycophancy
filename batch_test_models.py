#!/usr/bin/env python3
"""
批量测试所有Ollama模型的脚本
先用5条数据测试所有模型是否正常工作，然后运行全部数据
"""

import subprocess
import sys
import os
import time
import argparse
from datetime import datetime

# 要测试的模型列表（基于ollama list结果）
MODELS = [
    "rohithbojja/llava-med-v1.6:latest",
    "Adiptify/dentobot:latest",
    "llava:7b",
    "llava:13b",
    "qwen2.5vl:3b",
    "qwen2.5vl:7b", 
    "qwen2.5vl:32b",
    "gemma3:4b",
    "gemma3:12b",
    "gemma3:27b",
    "llama3.2-vision:11b",
    "qwen2.5vl:72b",
]

def run_test(model, samples, output_suffix=""):
    """运行单个模型的测试"""
    try:
        print(f"\n{'='*60}")
        print(f"开始测试模型: {model}")
        print(f"样本数量: {samples}")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        # 构建命令
        cmd = [
            "python3", 
            "test_ollama_unified.py",
            "--model", model,
            "--samples", str(samples)
        ]
        
        if output_suffix:
            model_clean = model.replace(":", "_").replace(".", "_").replace("/", "_")
            output_file = f"{model_clean}_{output_suffix}_results.csv"
            cmd.extend(["--output", output_file])
        
        print(f"执行命令: {' '.join(cmd)}")
        
        # 运行测试
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✅ 模型 {model} 测试成功!")
            print(f"⏱️ 耗时: {end_time - start_time:.2f} 秒")
            return True
        else:
            print(f"❌ 模型 {model} 测试失败!")
            print(f"错误输出: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 运行模型 {model} 时发生异常: {str(e)}")
        return False

def test_all_models_quick():
    """快速测试所有模型（5条数据）"""
    print("🚀 开始快速测试所有模型（5条数据）...")
    
    successful_models = []
    failed_models = []
    
    for model in MODELS:
        success = run_test(model, 5, "quick_test")
        if success:
            successful_models.append(model)
        else:
            failed_models.append(model)
        
        # 每个模型测试后短暂休息
        time.sleep(2)
    
    print(f"\n{'='*60}")
    print("快速测试结果汇总:")
    print(f"✅ 成功的模型 ({len(successful_models)}):")
    for model in successful_models:
        print(f"  - {model}")
    
    if failed_models:
        print(f"\n❌ 失败的模型 ({len(failed_models)}):")
        for model in failed_models:
            print(f"  - {model}")
    
    print(f"{'='*60}")
    
    return successful_models, failed_models

def test_all_models_full(successful_models):
    """对成功的模型运行全量测试"""
    if not successful_models:
        print("❌ 没有成功的模型，无法进行全量测试")
        return
    
    print(f"\n🎯 开始全量测试 {len(successful_models)} 个成功的模型...")
    
    for i, model in enumerate(successful_models, 1):
        print(f"\n进度: {i}/{len(successful_models)}")
        run_test(model, "all", "full")
        
        # 每个模型之间休息一下
        if i < len(successful_models):
            print("休息10秒...")
            time.sleep(10)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="批量测试所有Ollama模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python batch_test_models.py --mode quick    # 快速测试（5条数据）
  python batch_test_models.py --mode full     # 全量测试（所有数据）
  python batch_test_models.py --mode both     # 先快速测试，再全量测试
        """
    )
    
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['quick', 'full', 'both'],
        default='both',
        help='测试模式: quick=快速测试(5条), full=全量测试(所有), both=先快速再全量 (默认: both)'
    )
    
    args = parser.parse_args()
    
    print("🔬 Ollama模型批量测试工具")
    print(f"计划测试 {len(MODELS)} 个模型")
    print(f"测试模式: {args.mode}")
    
    # 检查是否存在必要文件
    if not os.path.exists("test_ollama_unified.py"):
        print("❌ 未找到 test_ollama_unified.py 文件!")
        sys.exit(1)
    
    if not os.path.exists("merged_dataset_with_question_type_cleaned.csv"):
        print("❌ 未找到数据集文件!")
        sys.exit(1)
    
    if args.mode == 'quick':
        print("🚀 执行快速测试模式...")
        successful_models, failed_models = test_all_models_quick()
        
    elif args.mode == 'full':
        print("🎯 执行全量测试模式...")
        # 直接对所有模型进行全量测试
        successful_models = []
        failed_models = []
        
        for i, model in enumerate(MODELS, 1):
            print(f"\n进度: {i}/{len(MODELS)}")
            success = run_test(model, "all", "full")
            if success:
                successful_models.append(model)
            else:
                failed_models.append(model)
            
            # 每个模型之间休息一下
            if i < len(MODELS):
                print("休息10秒...")
                time.sleep(10)
                
    else:  # both模式
        print("🔄 执行完整测试流程（先快速后全量）...")
        # 第一阶段：快速测试
        successful_models, failed_models = test_all_models_quick()
        
        if not successful_models:
            print("❌ 所有模型都测试失败，程序退出")
            sys.exit(1)
        
        # 第二阶段：全量测试成功的模型
        print(f"\n✨ 快速测试完成! {len(successful_models)} 个模型测试成功")
        print("🎯 开始全量测试...")
        test_all_models_full(successful_models)
    
    print(f"\n📊 最终结果:")
    print(f"✅ 成功: {len(successful_models)} 个模型")
    print(f"❌ 失败: {len(failed_models)} 个模型")
    
    if successful_models:
        print(f"\n✅ 成功的模型:")
        for model in successful_models:
            print(f"  - {model}")
    
    if failed_models:
        print(f"\n❌ 失败的模型:")
        for model in failed_models:
            print(f"  - {model}")
    
    print("\n🎉 所有测试完成!")

if __name__ == "__main__":
    main()
