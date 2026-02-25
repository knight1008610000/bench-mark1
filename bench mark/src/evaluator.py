"""
集成式评估器 - 包含API客户端、评估逻辑和结果生成功能
"""

import aiohttp
import asyncio
import json
import time
import os
import numpy as np
import yaml
from typing import Dict, List, Any, Optional
from scipy.special import comb
from collections import Counter
import datetime


class IntegratedEvaluator:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.problems = {}
        self.session = None
        self.start_time = time.time()
        self.completed_tasks = 0

    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件未找到: {config_path}")

    def _load_humaneval_data(self, data_path: str) -> Dict[str, Dict]:
        """加载HumanEval数据集"""
        problems = {}
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    problem_data = json.loads(line.strip())
                    problems[problem_data["task_id"]] = problem_data
        except FileNotFoundError:
            raise FileNotFoundError(f"数据文件未找到: {data_path}")

        print(f"✅ 成功加载 {len(problems)} 个HumanEval任务")
        return problems

    async def _init_session(self):
        """初始化HTTP会话"""
        self.session = aiohttp.ClientSession(
            headers={
                'Authorization': f'Bearer {self.config["model"]["api_key"]}',
                'Content-Type': 'application/json'
            }
        )
        print("✅ DeepSeek客户端初始化完成")

    async def _generate_code(self, prompt: str) -> Optional[str]:
        """调用DeepSeek API生成代码"""
        max_retries = self.config['evaluation'].get('max_retries', 3)

        for attempt in range(max_retries):
            try:
                payload = {
                    'model': self.config['model']['name'],
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': self.config['evaluation'].get('temperature', 0.1),
                    'max_tokens': self.config['evaluation'].get('max_tokens', 1024)
                }

                async with self.session.post(
                        f"{self.config['model']['base_url']}/chat/completions",
                        json=payload,
                        timeout=30
                ) as response:
                    result = await response.json()
                    return result['choices'][0]['message']['content']

            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"❌ API请求失败: {e}")
                    return None
                await asyncio.sleep(2 ** attempt)

        return None

    def _extract_code_from_response(self, response: str) -> str:
        """从模型响应中提取代码"""
        lines = response.split('\n')
        code_lines = []
        in_code_block = False

        for line in lines:
            if '```' in line:
                in_code_block = not in_code_block
                continue
            if in_code_block or line.strip().startswith('def ') or line.strip().startswith('class '):
                code_lines.append(line)

        return '\n'.join(code_lines).strip()

    def _safe_execute_test(self, problem: Dict, generated_code: str) -> Dict[str, Any]:
        """安全执行测试（模拟实现）"""
        # 在实际环境中，这里应该使用Docker沙箱
        import random
        return {
            'passed': random.random() > 0.7,  # 70%的通过率用于演示
            'execution_time': random.uniform(0.1, 2.0),
            'error': None if random.random() > 0.7 else {
                'error_type': 'AssertionError',
                'message': 'Test failed'
            }
        }

    def _compute_pass_at_k(self, n: int, c: int, k: int) -> float:
        """计算pass@k指标"""
        if n < k:
            return 0.0
        if n - c < k:
            return 1.0
        return 1.0 - comb(n - c, k, exact=True) / comb(n, k, exact=True)

    def _calculate_statistics(self, all_results: List[List[bool]]) -> Dict[str, Any]:
        """计算统计信息"""
        task_stats = []
        for task_results in all_results:
            n = len(task_results)
            c = sum(task_results)
            task_stats.append({
                'samples': n,
                'passed': c,
                'pass_rate': c / n if n > 0 else 0
            })

        pass_rates = [stat['pass_rate'] for stat in task_stats]
        total_samples = sum(stat['samples'] for stat in task_stats)
        total_passed = sum(stat['passed'] for stat in task_stats)

        return {
            'total_tasks': len(all_results),
            'total_samples': total_samples,
            'total_passed': total_passed,
            'overall_pass_rate': total_passed / total_samples if total_samples > 0 else 0,
            'avg_pass_rate': np.mean(pass_rates) if pass_rates else 0,
            'min_pass_rate': np.min(pass_rates) if pass_rates else 0,
            'max_pass_rate': np.max(pass_rates) if pass_rates else 0,
        }

    def _format_duration(self, seconds: float) -> str:
        """格式化时间显示"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def _estimate_remaining_time(self, completed: int, total: int) -> str:
        """估算剩余时间"""
        if completed == 0:
            return "计算中..."

        elapsed = time.time() - self.start_time
        time_per_task = elapsed / completed
        remaining_tasks = total - completed
        remaining_time = time_per_task * remaining_tasks

        return self._format_duration(remaining_time)

    def _create_progress_bar(self, completed: int, total: int, length: int = 30) -> str:
        """创建文本进度条"""
        if total == 0:
            return "[░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 0%"

        progress = completed / total
        filled_length = int(length * progress)
        bar = '█' * filled_length + '░' * (length - filled_length)
        percentage = int(progress * 100)

        return f"[{bar}] {percentage}%"

    def _generate_prompt(self, problem: Dict) -> str:
        """生成评估提示词"""
        return f"""请完成以下Python函数：

{problem['prompt']}

要求：
1. 只返回完整的函数实现代码
2. 不要包含任何额外的解释或注释
3. 确保代码语法正确"""

    async def _evaluate_single_task(self, task_id: str, problem: Dict) -> Dict[str, Any]:
        """评估单个任务"""
        num_samples = self.config['evaluation']['num_samples_per_task']
        results = []
        generated_codes = []
        test_results = []

        for i in range(num_samples):
            try:
                # 生成代码
                prompt = self._generate_prompt(problem)
                response = await self._generate_code(prompt)

                if response is None:
                    results.append(False)
                    generated_codes.append("")
                    test_results.append({'passed': False, 'error': 'API请求失败'})
                    continue

                # 提取代码
                generated_code = self._extract_code_from_response(response)
                generated_codes.append(generated_code)

                # 执行测试
                test_result = self._safe_execute_test(problem, generated_code)
                test_results.append(test_result)
                results.append(test_result['passed'])

                # 添加请求延迟
                await asyncio.sleep(self.config['evaluation'].get('request_delay', 1.0))

            except Exception as e:
                print(f"❌ 任务 {task_id} 样本 {i} 出错: {e}")
                results.append(False)
                generated_codes.append("")
                test_results.append({'passed': False, 'error': str(e)})

        self.completed_tasks += 1
        return {
            'task_id': task_id,
            'results': results,
            'generated_codes': generated_codes,
            'test_results': test_results
        }

    def _generate_filename(self) -> str:
        """生成结果文件名"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.config['model']['name']
        return f"{model_name}_humaneval_{timestamp}.json"

    def _create_result_data(self, evaluation_results: Dict, duration: float) -> Dict[str, Any]:
        """创建完整的结果数据"""
        stats = self._calculate_statistics(evaluation_results['all_results'])

        return {
            'evaluation_info': {
                'model_name': self.config['model']['name'],
                'evaluation_date': datetime.datetime.now().isoformat(),
                'dataset': 'HumanEval',
                'total_tasks': stats['total_tasks'],
                'evaluated_tasks': stats['total_tasks'],
                'evaluation_duration': self._format_duration(duration)
            },
            'evaluation_parameters': self.config['evaluation'],
            'summary_metrics': {
                'pass@1': self._compute_pass_at_k(stats['total_samples'], stats['total_passed'], 1),
                'pass@10': self._compute_pass_at_k(stats['total_samples'], stats['total_passed'], 10),
                'pass@100': self._compute_pass_at_k(stats['total_samples'], stats['total_passed'], 100),
                'total_passed_samples': stats['total_passed'],
                'total_failed_samples': stats['total_samples'] - stats['total_passed'],
                'overall_pass_rate': stats['overall_pass_rate']
            },
            'detailed_results': self._create_detailed_results(evaluation_results),
            'system_info': {
                'python_version': '3.8+',
                'evaluation_framework': '联想小天集成评估系统'
            }
        }

    def _create_detailed_results(self, evaluation_results: Dict) -> List[Dict]:
        """创建详细结果"""
        detailed = []

        for i, (task_id, task_results) in enumerate(zip(
                evaluation_results['problems'].keys(),
                evaluation_results['all_results']
        )):
            n = len(task_results)
            c = sum(task_results)

            detailed.append({
                'task_id': task_id,
                'samples_count': n,
                'passed_count': c,
                'pass_rate': c / n if n > 0 else 0,
                'pass@1': self._compute_pass_at_k(n, c, 1),
                'pass@10': self._compute_pass_at_k(n, c, 10)
            })

        return detailed

    def _write_results(self, evaluation_results: Dict, duration: float) -> str:
        """写入结果文件"""
        result_data = self._create_result_data(evaluation_results, duration)
        output_dir = self.config['output']['directory']
        os.makedirs(output_dir, exist_ok=True)

        filename = self._generate_filename()
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        return filepath

    async def evaluate_all_tasks(self) -> Dict[str, Any]:
        """评估所有任务"""
        # 加载数据
        self.problems = self._load_humaneval_data(self.config['evaluation']['data_path'])

        # 初始化会话
        await self._init_session()

        max_tasks = self.config['evaluation'].get('max_tasks', len(self.problems))
        task_ids = list(self.problems.keys())[:max_tasks]
        total_tasks = len(task_ids)

        print(f"🚀 开始评估 {total_tasks} 个任务，每个任务 {self.config['evaluation']['num_samples_per_task']} 个样本")

        all_results = []
        all_generated_codes = []
        all_test_results = []

        # 使用信号量控制并发数
        semaphore = asyncio.Semaphore(3)

        async def limited_evaluate(task_id):
            async with semaphore:
                return await self._evaluate_single_task(task_id, self.problems[task_id])

        tasks = [limited_evaluate(task_id) for task_id in task_ids]

        for i, task in enumerate(asyncio.as_completed(tasks)):
            result = await task
            all_results.append(result['results'])
            all_generated_codes.append(result['generated_codes'])
            all_test_results.append(result['test_results'])

            # 更新进度
            elapsed = time.time() - self.start_time
            remaining = self._estimate_remaining_time(i + 1, total_tasks)
            progress_bar = self._create_progress_bar(i + 1, total_tasks)
            print(f"\r{progress_bar} | 已用: {self._format_duration(elapsed)} | 剩余: {remaining}", end="")

        print("\n✅ 所有任务评估完成！")

        return {
            'all_results': all_results,
            'all_generated_codes': all_generated_codes,
            'all_test_results': all_test_results,
            'problems': {k: v for k, v in self.problems.items() if k in task_ids}
        }

    async def run_evaluation(self):
        """运行完整的评估流程"""
        try:
            # 执行评估
            start_time = time.time()
            evaluation_results = await self.evaluate_all_tasks()
            duration = time.time() - start_time

            # 写入结果
            result_file = self._write_results(evaluation_results, duration)

            print(f"✅ 评估完成！结果已保存至: {result_file}")
            print(f"⏱️  总耗时: {duration:.1f} 秒")

            # 显示汇总结果
            stats = self._calculate_statistics(evaluation_results['all_results'])
            print(f"📊 汇总结果: {stats['total_passed']}/{stats['total_samples']} 样本通过")
            print(f"🎯 总体通过率: {stats['overall_pass_rate']:.3f}")

            return result_file

        except Exception as e:
            print(f"❌ 评估过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def close(self):
        """清理资源"""
        if self.session:
            await self.session.close()
            print("✅ 会话已关闭")
