#!/usr/bin/env python3
"""
HumanEval DeepSeek 评估系统 - 简化主程序
"""

import asyncio
from src.evaluator import IntegratedEvaluator


async def main():
    print("🚀 启动 HumanEval DeepSeek 集成评估系统")
    print("=" * 50)

    # 创建评估器并运行
    evaluator = IntegratedEvaluator()
    result_file = await evaluator.run_evaluation()

    # 清理资源
    await evaluator.close()

    if result_file:
        print(f"🎉 评估成功完成！结果文件: {result_file}")
    else:
        print("❌ 评估失败")


if __name__ == "__main__":
    asyncio.run(main())
