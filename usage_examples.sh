#!/bin/bash
# OpenAI API测试工具使用示例

# 设置API密钥
API_KEY="sk-m6RCv7RNcBb5379c5412T3BlbKFJA6AC4d65Aace4AA5939e"

# 数据集路径
DATASET="/home/zikun/github/Sycophency/NC_Scyphancy/NC_Challenge_benchmark/sycophancy_challenge_set.csv"

echo "=== OpenAI API 谄媚性测试工具使用示例 ==="
echo ""

# 示例1: 测试模式 - 只测试5条数据
echo "1. 测试模式（5条数据，估算成本）:"
echo "python openai_api_tester.py --api-key $API_KEY --dataset $DATASET --test-mode --test-count 5"
echo ""

# 示例2: 完整测试 - 单进程模式
echo "2. 完整测试 - 单进程模式:"
echo "python openai_api_tester.py --api-key $API_KEY --dataset $DATASET"
echo ""

# 示例3: 完整测试 - 多进程模式
echo "3. 完整测试 - 多进程模式（3个线程）:"
echo "python openai_api_tester.py --api-key $API_KEY --dataset $DATASET --multiprocess --max-workers 3"
echo ""

# 示例4: 自定义配置
echo "4. 自定义配置:"
echo "python openai_api_tester.py \\"
echo "    --api-key $API_KEY \\"
echo "    --dataset $DATASET \\"
echo "    --model gpt-4o \\"
echo "    --base-url https://aigptx.top/v1 \\"
echo "    --image-folder challenge_dataset \\"
echo "    --output-dir api_results \\"
echo "    --multiprocess \\"
echo "    --max-workers 5"
echo ""

# 示例5: 禁用断点续传
echo "5. 从头开始测试（禁用断点续传）:"
echo "python openai_api_tester.py --api-key $API_KEY --dataset $DATASET --no-resume"
echo ""

echo "=== 参数说明 ==="
echo "--api-key        : OpenAI API密钥（必需）"
echo "--dataset        : 数据集CSV文件路径（必需）"
echo "--base-url       : API基础URL（默认: https://aigptx.top/v1）"
echo "--model          : 模型名称（默认: gpt-4o）"
echo "--image-folder   : 图片文件夹路径（默认: challenge_dataset）"
echo "--output-dir     : 输出结果目录（默认: api_results）"
echo "--test-mode      : 启用测试模式（只测试少量数据）"
echo "--test-count     : 测试模式下测试的数据条数（默认: 5）"
echo "--multiprocess   : 启用多进程模式"
echo "--max-workers    : 最大并发线程数（默认: 3）"
echo "--no-resume      : 禁用断点续传"
echo ""

echo "建议使用流程:"
echo "1. 先运行测试模式估算成本: python openai_api_tester.py --api-key $API_KEY --dataset $DATASET --test-mode"
echo "2. 如果成本可接受，再运行完整测试: python openai_api_tester.py --api-key $API_KEY --dataset $DATASET --multiprocess"
