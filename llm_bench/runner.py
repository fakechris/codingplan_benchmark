"""
异步测试 Runner

支持:
1. 单次请求测试 (质量 + 速度)
2. 并发吞吐测试
3. 一致性测试 (多次运行同一任务)
4. 完整 benchmark 流程编排
5. Per-model 参数 (temperature, max_tokens 等独立配置)
"""

from __future__ import annotations

import asyncio
import time
import json
import random
import string
from dataclasses import dataclass, field
from typing import Optional, Callable

from .providers.base import Provider, CompletionResult
from .tasks.coding_plans import (
    Task, TaskDifficulty,
    get_tasks_by_difficulty, get_speed_tasks, get_throughput_tasks,
    get_consistency_tasks, get_trap_tasks, get_debug_tasks, TASKS,
)
from .scorer import RuleScorer, JudgeScorer, QualityScore, compute_final_score
from .config import ModelConfig


@dataclass
class TaskResult:
    """单个任务的测试结果"""
    task: Task
    completion: CompletionResult
    quality: QualityScore

    @property
    def summary(self) -> dict:
        c = self.completion
        return {
            "task_id": self.task.id,
            "task_title": self.task.title,
            "difficulty": self.task.difficulty.value,
            "category": self.task.category.value,
            # 客观指标
            "ttft_s": round(c.ttft, 3),
            "gen_time_s": round(c.gen_time, 3),
            "total_latency_s": round(c.total_latency, 3),
            "tps": round(c.tps, 1),
            "chars_per_second": round(c.chars_per_second, 1),
            "completion_tokens": c.completion_tokens,
            "prompt_tokens": c.prompt_tokens,
            "output_chars": c.output_chars,
            # 质量指标
            "quality_score": round(self.quality.final_score, 1),
            "rule_score": round(self.quality.rule_score, 1),
            "judge_score": round(self.quality.judge_score, 1),
            "error": c.error,
        }


@dataclass
class ThroughputResult:
    concurrency: int
    total_requests: int
    successful: int
    failed: int
    total_time: float
    avg_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    requests_per_second: float
    avg_tps: float
    errors: list[str] = field(default_factory=list)


@dataclass
class ConsistencyResult:
    task_id: str
    num_runs: int
    scores: list[float]
    avg_score: float
    score_stddev: float
    responses_similarity: float


@dataclass
class BenchmarkResult:
    """完整的 benchmark 结果"""
    model: str
    provider: str
    timestamp: str
    model_config_name: str = ""
    task_results: list[TaskResult] = field(default_factory=list)
    throughput_result: Optional[ThroughputResult] = None
    consistency_results: list[ConsistencyResult] = field(default_factory=list)

    quality_score: float = 0.0
    speed_score: float = 0.0
    throughput_score: float = 0.0
    consistency_score: float = 0.0
    overall_score: float = 0.0

    # 整体耗时
    wall_time: float = 0.0  # 该模型完整评测的挂钟时间 (秒)

    def get_objective_metrics(self) -> dict:
        """计算聚合客观指标"""
        valid = [r for r in self.task_results if not r.completion.error]
        if not valid:
            return {}

        ttfts = [r.completion.ttft for r in valid if r.completion.ttft > 0]
        tps_list = [r.completion.tps for r in valid if r.completion.tps > 0]
        lats = [r.completion.total_latency for r in valid if r.completion.total_latency > 0]
        gen_times = [r.completion.gen_time for r in valid if r.completion.gen_time > 0]
        cps_list = [r.completion.chars_per_second for r in valid if r.completion.chars_per_second > 0]
        out_tokens = [r.completion.completion_tokens for r in valid]
        out_chars = [r.completion.output_chars for r in valid]

        def _stats(lst):
            if not lst:
                return {"min": 0, "max": 0, "avg": 0, "p50": 0}
            s = sorted(lst)
            return {
                "min": s[0],
                "max": s[-1],
                "avg": sum(s) / len(s),
                "p50": s[len(s) // 2],
            }

        return {
            "task_count": len(valid),
            "success_rate": len(valid) / max(1, len(self.task_results)) * 100,
            "wall_time": self.wall_time,
            "total_output_tokens": sum(out_tokens),
            "total_output_chars": sum(out_chars),
            "ttft": _stats(ttfts),
            "tps": _stats(tps_list),
            "latency": _stats(lats),
            "gen_time": _stats(gen_times),
            "cps": _stats(cps_list),
        }

    def compute_overall(self):
        if self.task_results:
            valid = [r for r in self.task_results if not r.completion.error]
            if valid:
                self.quality_score = sum(r.quality.final_score for r in valid) / len(valid)

        if self.task_results:
            valid = [r for r in self.task_results if not r.completion.error and r.completion.tps > 0]
            if valid:
                import math
                avg_ttft = sum(r.completion.ttft for r in valid) / len(valid)
                avg_tps = sum(r.completion.tps for r in valid) / len(valid)
                ttft_score = max(0, min(100, 100 - (math.log(max(0.1, avg_ttft)) + 0.7) * 40))
                tps_score = max(0, min(100, (avg_tps - 5) / 95 * 100))
                self.speed_score = ttft_score * 0.4 + tps_score * 0.6

        if self.throughput_result:
            tr = self.throughput_result
            success_rate = tr.successful / max(1, tr.total_requests) * 100
            rps_score = min(100, tr.requests_per_second / 5 * 100)
            self.throughput_score = success_rate * 0.5 + rps_score * 0.5

        if self.consistency_results:
            avg_stddev = sum(r.score_stddev for r in self.consistency_results) / len(self.consistency_results)
            self.consistency_score = max(0, min(100, 100 - (avg_stddev - 5) / 25 * 100))

        weights = {"quality": 0.45, "speed": 0.25, "throughput": 0.15, "consistency": 0.15}
        self.overall_score = (
            self.quality_score * weights["quality"]
            + self.speed_score * weights["speed"]
            + self.throughput_score * weights["throughput"]
            + self.consistency_score * weights["consistency"]
        )

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "model_name": self.model_config_name,
            "provider": self.provider,
            "timestamp": self.timestamp,
            "wall_time_s": round(self.wall_time, 2),
            "scores": {
                "overall": round(self.overall_score, 1),
                "quality": round(self.quality_score, 1),
                "speed": round(self.speed_score, 1),
                "throughput": round(self.throughput_score, 1),
                "consistency": round(self.consistency_score, 1),
            },
            "objective_metrics": self.get_objective_metrics(),
            "tasks": [r.summary for r in self.task_results],
            "throughput": {
                "concurrency": self.throughput_result.concurrency,
                "rps": round(self.throughput_result.requests_per_second, 2),
                "success_rate": round(self.throughput_result.successful / max(1, self.throughput_result.total_requests) * 100, 1),
                "p50_latency": round(self.throughput_result.p50_latency, 3),
                "p95_latency": round(self.throughput_result.p95_latency, 3),
                "avg_tps": round(self.throughput_result.avg_tps, 1),
            } if self.throughput_result else None,
            "consistency": [
                {
                    "task_id": r.task_id,
                    "avg_score": round(r.avg_score, 1),
                    "stddev": round(r.score_stddev, 2),
                    "similarity": round(r.responses_similarity, 2),
                }
                for r in self.consistency_results
            ],
        }


class BenchmarkRunner:
    """Benchmark 运行器"""

    def __init__(
        self,
        provider: Provider,
        model_config: Optional[ModelConfig] = None,
        judge_provider: Optional[Provider] = None,
        global_temperature: float = 0.0,
        concurrency: int = 5,
        consistency_runs: int = 3,
        enable_judge: bool = True,
        task_filter: Optional[list[str]] = None,
        difficulty_filter: Optional[TaskDifficulty] = None,
        progress_callback: Optional[Callable] = None,
        throughput_multiplier: int = 3,
        max_quality_tasks: int = 0,
        anti_cache: bool = False,
    ):
        self.provider = provider
        self.model_config = model_config
        self.judge_provider = judge_provider
        self.global_temperature = global_temperature
        self.concurrency = concurrency
        self.consistency_runs = consistency_runs
        self.enable_judge = enable_judge
        self.task_filter = task_filter
        self.difficulty_filter = difficulty_filter
        self.progress_callback = progress_callback
        self.throughput_multiplier = throughput_multiplier
        self.max_quality_tasks = max_quality_tasks
        self.anti_cache = anti_cache

        self.rule_scorer = RuleScorer()
        self.judge_scorer = JudgeScorer()

    def _get_temperature(self) -> float:
        if self.model_config and self.model_config.temperature is not None:
            return self.model_config.temperature
        return self.global_temperature

    def _get_max_tokens(self, task: Task) -> int:
        if self.model_config and self.model_config.max_tokens is not None:
            return self.model_config.max_tokens
        return task.max_tokens

    def _get_system_prompt(self, task: Task) -> str:
        if self.model_config and self.model_config.system_prompt_override:
            return self.model_config.system_prompt_override
        return task.system_prompt

    def _get_tasks(self) -> list[Task]:
        tasks = TASKS
        if self.difficulty_filter:
            tasks = get_tasks_by_difficulty(self.difficulty_filter)
        if self.task_filter:
            tasks = [t for t in tasks if t.id in self.task_filter]
        # 排除 debug 专用任务 (除非明确指定)
        if not self.task_filter:
            tasks = [t for t in tasks if "debug" not in t.tags]
        # 限制质量题数量
        if self.max_quality_tasks > 0 and len(tasks) > self.max_quality_tasks:
            tasks = tasks[:self.max_quality_tasks]
        return tasks

    @staticmethod
    def _make_nonce() -> str:
        """生成随机 nonce 附加到 prompt, 用于击穿 provider 缓存"""
        rid = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        return f"\n\n[session-id: {rid}]"

    def _maybe_add_nonce(self, prompt: str) -> str:
        if self.anti_cache:
            return prompt + self._make_nonce()
        return prompt

    async def run_single_task(self, task: Task) -> TaskResult:
        """运行单个任务 (带重试)"""
        retry_count = self.model_config.retry_count if self.model_config else 2
        retry_delay = self.model_config.retry_delay if self.model_config else 1.0

        last_result = None
        for attempt in range(retry_count + 1):
            if attempt > 0:
                await asyncio.sleep(retry_delay)

            messages = []
            sys_prompt = self._get_system_prompt(task)
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            messages.append({"role": "user", "content": self._maybe_add_nonce(task.prompt)})

            completion = await self.provider.complete(
                messages=messages,
                temperature=self._get_temperature(),
                max_tokens=self._get_max_tokens(task),
            )

            if not completion.error:
                quality = self.rule_scorer.score(task, completion.text)

                if self.enable_judge and self.judge_provider:
                    try:
                        judge_messages = self.judge_scorer.build_judge_messages(task, completion.text)
                        judge_result = await self.judge_provider.complete(
                            messages=judge_messages, temperature=0.0, max_tokens=512, stream=False,
                        )
                        if not judge_result.error:
                            judge_score, feedback = self.judge_scorer.parse_judge_response(judge_result.text)
                            quality = compute_final_score(quality, judge_score, feedback)
                    except Exception as e:
                        quality.judge_feedback = f"Judge error: {e}"

                return TaskResult(task=task, completion=completion, quality=quality)

            last_result = completion

        # 全部重试失败
        quality = QualityScore(task_id=task.id)
        return TaskResult(task=task, completion=last_result or CompletionResult(text="", model=self.provider.model, error="all retries failed"), quality=quality)

    async def run_debug_task(self) -> TaskResult:
        """运行 debug 任务 (D01, 连通性测试)"""
        debug_tasks = get_debug_tasks()
        task = debug_tasks[0] if debug_tasks else TASKS[0]
        return await self.run_single_task(task)

    async def run_throughput_test(self) -> ThroughputResult:
        speed_tasks = get_speed_tasks()
        task = speed_tasks[0] if speed_tasks else [t for t in TASKS if t.difficulty == TaskDifficulty.EASY][0]

        total_requests = self.concurrency * self.throughput_multiplier
        latencies = []
        tps_list = []
        errors = []

        semaphore = asyncio.Semaphore(self.concurrency)

        def _build_messages():
            msgs = []
            if task.system_prompt:
                msgs.append({"role": "system", "content": task.system_prompt})
            msgs.append({"role": "user", "content": self._maybe_add_nonce(task.prompt)})
            return msgs

        async def single_request():
            async with semaphore:
                return await self.provider.complete(
                    messages=_build_messages(), temperature=0.1, max_tokens=task.max_tokens,
                )

        start = time.time()
        results = await asyncio.gather(*[single_request() for _ in range(total_requests)], return_exceptions=True)
        total_time = time.time() - start

        for r in results:
            if isinstance(r, Exception):
                errors.append(str(r))
            elif isinstance(r, CompletionResult):
                if r.error:
                    errors.append(r.error)
                else:
                    latencies.append(r.total_latency)
                    if r.tps > 0:
                        tps_list.append(r.tps)

        latencies.sort()

        def pct(lst, p):
            if not lst:
                return 0
            k = (len(lst) - 1) * p / 100
            f = int(k)
            c = min(f + 1, len(lst) - 1)
            return lst[f] + (k - f) * (lst[c] - lst[f])

        return ThroughputResult(
            concurrency=self.concurrency, total_requests=total_requests,
            successful=len(latencies), failed=len(errors), total_time=total_time,
            avg_latency=sum(latencies) / max(1, len(latencies)),
            p50_latency=pct(latencies, 50), p95_latency=pct(latencies, 95), p99_latency=pct(latencies, 99),
            requests_per_second=len(latencies) / max(0.1, total_time),
            avg_tps=sum(tps_list) / max(1, len(tps_list)),
            errors=errors[:10],
        )

    async def run_consistency_test(self) -> list[ConsistencyResult]:
        consistency_tasks = get_consistency_tasks()
        if not consistency_tasks:
            consistency_tasks = get_trap_tasks()[:2]

        results = []
        for task in consistency_tasks:
            scores = []
            responses = []
            for _ in range(self.consistency_runs):
                tr = await self.run_single_task(task)
                scores.append(tr.quality.rule_score)
                responses.append(tr.completion.text)

            import numpy as np
            arr = np.array(scores)
            similarity = self._compute_similarity(responses)

            results.append(ConsistencyResult(
                task_id=task.id, num_runs=self.consistency_runs, scores=scores,
                avg_score=float(arr.mean()), score_stddev=float(arr.std()),
                responses_similarity=similarity,
            ))
        return results

    def _compute_similarity(self, texts: list[str]) -> float:
        if len(texts) < 2:
            return 1.0

        def jaccard(a, b):
            wa, wb = set(a.lower().split()), set(b.lower().split())
            return len(wa & wb) / max(1, len(wa | wb))

        sims = [jaccard(texts[i], texts[j]) for i in range(len(texts)) for j in range(i+1, len(texts))]
        return sum(sims) / max(1, len(sims))

    def _report(self, phase, current, total, detail=""):
        if self.progress_callback:
            self.progress_callback(phase, current, total, detail)

    async def run_full_benchmark(self) -> BenchmarkResult:
        from datetime import datetime
        name = self.model_config.name if self.model_config else self.provider.model

        wall_start = time.time()

        result = BenchmarkResult(
            model=self.provider.model, provider=self.provider.__class__.__name__,
            timestamp=datetime.now().isoformat(), model_config_name=name,
        )

        tasks = self._get_tasks()

        self._report("quality", 0, len(tasks), "开始 coding 能力评测...")
        for i, task in enumerate(tasks):
            self._report("quality", i + 1, len(tasks), f"[{task.id}] {task.title}")
            try:
                tr = await self.run_single_task(task)
                result.task_results.append(tr)
            except Exception as e:
                comp = CompletionResult(text="", model=self.provider.model, error=str(e))
                result.task_results.append(TaskResult(task=task, completion=comp, quality=QualityScore(task_id=task.id)))

        self._report("throughput", 0, 1, "开始吞吐测试...")
        try:
            result.throughput_result = await self.run_throughput_test()
            self._report("throughput", 1, 1, "完成")
        except Exception as e:
            self._report("throughput", 1, 1, f"失败: {e}")

        self._report("consistency", 0, 1, "开始一致性测试...")
        try:
            result.consistency_results = await self.run_consistency_test()
            self._report("consistency", 1, 1, "完成")
        except Exception as e:
            self._report("consistency", 1, 1, f"失败: {e}")

        result.wall_time = time.time() - wall_start
        result.compute_overall()
        return result
