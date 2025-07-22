#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
线程数性能测试工具
用于找到最佳的并发线程数配置
"""

import time
import subprocess
import os
import json
import argparse
from datetime import datetime


def test_thread_performance(api_key, dataset_path, image_folder, model="gpt-4o", 
                          base_url="https://c-z0-api-01.hash070.com/v1", 
                          test_counts=[5, 10], thread_counts=[3, 5, 10, 15, 20]):
    """测试不同线程数的性能"""
    
    results = []
    
    print(f"开始线程数性能测试")
    print(f"模型: {model}")
    print(f"测试线程数: {thread_counts}")
    print(f"每个配置测试样本数: {test_counts}")
    print("="*60)
    
    for test_count in test_counts:
        print(f"\n测试样本数: {test_count}")
        print("-" * 40)
        
        for thread_count in thread_counts:
            print(f"\n测试 {thread_count} 线程...")
            
            # 清理之前的测试文件
            output_dir = f"thread_test_{thread_count}"
            if os.path.exists(output_dir):
                subprocess.run(['rm', '-rf', output_dir], check=True)
            
            cmd = [
                'python', 'openai_api_tester.py',
                '--api-key', api_key,
                '--dataset', dataset_path,
                '--base-url', base_url,
                '--model', model,
                '--image-folder', image_folder,
                '--output-dir', output_dir,
                '--multiprocess',
                '--max-workers', str(thread_count),
                '--test-mode',
                '--test-count', str(test_count),
                '--no-resume'  # 禁用续传确保每次都是全新测试
            ]
            
            try:
                start_time = time.time()
                result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)  # 5分钟超时
                end_time = time.time()
                
                duration = end_time - start_time
                samples_per_minute = test_count / (duration / 60) if duration > 0 else 0
                
                # 尝试读取统计信息
                stats_file = os.path.join(output_dir, f"openai_{model.replace('-', '_')}_stats.json")
                tokens_used = 0
                api_calls = 0
                
                if os.path.exists(stats_file):
                    with open(stats_file, 'r') as f:
                        stats = json.load(f)
                        tokens_used = stats.get('total_tokens', 0)
                        api_calls = stats.get('total_api_calls', 0)
                
                result_data = {
                    'test_count': test_count,
                    'thread_count': thread_count,
                    'duration_seconds': duration,
                    'samples_per_minute': samples_per_minute,
                    'tokens_used': tokens_used,
                    'api_calls': api_calls,
                    'success': True,
                    'error': None
                }
                
                print(f"  ✅ 成功! 用时: {duration:.1f}s, 速度: {samples_per_minute:.1f} 样本/分钟")
                
            except subprocess.TimeoutExpired:
                result_data = {
                    'test_count': test_count,
                    'thread_count': thread_count,
                    'duration_seconds': 300,
                    'samples_per_minute': 0,
                    'tokens_used': 0,
                    'api_calls': 0,
                    'success': False,
                    'error': 'Timeout'
                }
                print(f"  ❌ 超时!")
                
            except subprocess.CalledProcessError as e:
                result_data = {
                    'test_count': test_count,
                    'thread_count': thread_count,
                    'duration_seconds': 0,
                    'samples_per_minute': 0,
                    'tokens_used': 0,
                    'api_calls': 0,
                    'success': False,
                    'error': str(e)
                }
                print(f"  ❌ 失败: {e}")
            
            results.append(result_data)
            
            # 清理测试文件
            if os.path.exists(output_dir):
                subprocess.run(['rm', '-rf', output_dir])
            
            # 等待一下避免API限制
            time.sleep(2)
    
    return results


def analyze_results(results):
    """分析测试结果"""
    print("\n" + "="*80)
    print("线程数性能测试结果分析")
    print("="*80)
    
    # 按测试样本数分组
    by_test_count = {}
    for result in results:
        test_count = result['test_count']
        if test_count not in by_test_count:
            by_test_count[test_count] = []
        by_test_count[test_count].append(result)
    
    for test_count, test_results in by_test_count.items():
        print(f"\n样本数 {test_count}:")
        print(f"{'线程数':<8} {'成功':<6} {'用时(s)':<10} {'速度(样本/分)':<12} {'Token':<8}")
        print("-" * 50)
        
        successful_results = [r for r in test_results if r['success']]
        
        for result in test_results:
            status = "✅" if result['success'] else "❌"
            duration = f"{result['duration_seconds']:.1f}" if result['success'] else "N/A"
            speed = f"{result['samples_per_minute']:.1f}" if result['success'] else "N/A"
            tokens = f"{result['tokens_used']}" if result['success'] else "N/A"
            
            print(f"{result['thread_count']:<8} {status:<6} {duration:<10} {speed:<12} {tokens:<8}")
        
        if successful_results:
            # 找到最佳线程数
            best_result = max(successful_results, key=lambda x: x['samples_per_minute'])
            print(f"\n🏆 最佳配置: {best_result['thread_count']} 线程")
            print(f"   速度: {best_result['samples_per_minute']:.1f} 样本/分钟")
            print(f"   用时: {best_result['duration_seconds']:.1f} 秒")
    
    # 整体建议
    all_successful = [r for r in results if r['success']]
    if all_successful:
        best_overall = max(all_successful, key=lambda x: x['samples_per_minute'])
        
        print(f"\n📊 整体最佳配置:")
        print(f"   线程数: {best_overall['thread_count']}")
        print(f"   样本数: {best_overall['test_count']}")
        print(f"   速度: {best_overall['samples_per_minute']:.1f} 样本/分钟")
        
        # 稳定性分析
        thread_success_rate = {}
        for result in results:
            thread_count = result['thread_count']
            if thread_count not in thread_success_rate:
                thread_success_rate[thread_count] = {'success': 0, 'total': 0}
            
            thread_success_rate[thread_count]['total'] += 1
            if result['success']:
                thread_success_rate[thread_count]['success'] += 1
        
        print(f"\n📈 稳定性分析:")
        print(f"{'线程数':<8} {'成功率':<8}")
        print("-" * 20)
        for thread_count, stats in sorted(thread_success_rate.items()):
            success_rate = stats['success'] / stats['total'] * 100
            print(f"{thread_count:<8} {success_rate:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='线程数性能测试工具')
    
    parser.add_argument('--api-key', required=True, help='API密钥')
    parser.add_argument('--dataset', required=True, help='数据集CSV文件路径')
    parser.add_argument('--image-folder', required=True, help='图片文件夹路径')
    parser.add_argument('--model', default='gpt-4o', help='使用的模型名称')
    parser.add_argument('--base-url', default='https://c-z0-api-01.hash070.com/v1', help='API基础URL')
    
    parser.add_argument('--thread-counts', nargs='*', type=int, default=[3, 5, 10, 15, 20],
                       help='要测试的线程数列表')
    parser.add_argument('--test-counts', nargs='*', type=int, default=[5, 10],
                       help='要测试的样本数列表')
    
    args = parser.parse_args()
    
    print("开始线程数性能测试...")
    print(f"这将测试 {len(args.thread_counts)} 种线程配置 × {len(args.test_counts)} 种样本数")
    print(f"预计需要 {len(args.thread_counts) * len(args.test_counts) * 2} 分钟")
    
    input("按Enter继续，或Ctrl+C取消...")
    
    results = test_thread_performance(
        api_key=args.api_key,
        dataset_path=args.dataset,
        image_folder=args.image_folder,
        model=args.model,
        base_url=args.base_url,
        test_counts=args.test_counts,
        thread_counts=args.thread_counts
    )
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"thread_performance_test_{timestamp}.json"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {results_file}")
    
    # 分析结果
    analyze_results(results)


if __name__ == '__main__':
    main()
