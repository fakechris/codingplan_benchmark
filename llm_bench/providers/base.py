"""
Provider 基类 - 定义统一的 LLM API 接口
支持流式输出以精确测量 TTFT 和 TPS
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional


@dataclass
class StreamChunk:
    """流式输出的单个片段"""
    text: str
    timestamp: float = field(default_factory=time.time)
    finish_reason: Optional[str] = None


@dataclass
class CompletionResult:
    """一次完整的补全结果, 包含性能指标"""
    text: str
    model: str

    # 时间指标
    request_start: float = 0.0
    first_token_time: float = 0.0   # 第一个 text token 的时间
    end_time: float = 0.0

    # Thinking 阶段指标 (reasoning 模型)
    thinking_start_time: float = 0.0   # thinking 阶段开始时间
    thinking_end_time: float = 0.0     # thinking 阶段结束时间
    thinking_tokens: int = 0           # thinking 消耗的 token 数
    thinking_text: str = ""            # thinking 内容 (调试用, 不计入质量评分)

    # Token 统计
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # 流式 chunk 时间戳 (用于计算 TPS 曲线)
    chunk_timestamps: list[float] = field(default_factory=list)
    chunk_texts: list[str] = field(default_factory=list)

    # 错误信息
    error: Optional[str] = None

    @property
    def ttft(self) -> float:
        """Time To First Token (秒)"""
        if self.first_token_time and self.request_start:
            return self.first_token_time - self.request_start
        return 0.0

    @property
    def total_latency(self) -> float:
        """总延迟 (秒)"""
        if self.end_time and self.request_start:
            return self.end_time - self.request_start
        return 0.0

    @property
    def thinking_time(self) -> float:
        """Thinking 阶段耗时 (秒), reasoning 模型专属"""
        if self.thinking_end_time and self.thinking_start_time:
            return self.thinking_end_time - self.thinking_start_time
        return 0.0

    @property
    def has_thinking(self) -> bool:
        """是否有 thinking 阶段"""
        return self.thinking_time > 0

    @property
    def gen_time(self) -> float:
        """纯生成时间 (去除 TTFT, 秒)"""
        gt = self.total_latency - self.ttft
        return gt if gt > 0 else 0.0

    @property
    def tps(self) -> float:
        """Tokens Per Second (基于 completion tokens)"""
        if self.gen_time > 0 and self.completion_tokens > 0:
            return self.completion_tokens / self.gen_time
        return 0.0

    @property
    def output_chars(self) -> int:
        """输出字符数"""
        return len(self.text) if self.text else 0

    @property
    def chars_per_second(self) -> float:
        """Characters Per Second"""
        if self.gen_time > 0 and self.output_chars > 0:
            return self.output_chars / self.gen_time
        return 0.0

    @property
    def tps_timeline(self) -> list[tuple[float, float]]:
        """TPS 时间线 - 每个 chunk 的瞬时 TPS"""
        if len(self.chunk_timestamps) < 2:
            return []
        timeline = []
        for i in range(1, len(self.chunk_timestamps)):
            dt = self.chunk_timestamps[i] - self.chunk_timestamps[i - 1]
            if dt > 0:
                chars = len(self.chunk_texts[i])
                # 粗略估计 tokens ≈ chars / 3.5 (中英文混合)
                est_tokens = max(1, chars / 3.5)
                timeline.append((self.chunk_timestamps[i] - self.request_start, est_tokens / dt))
        return timeline


class Provider(ABC):
    """LLM Provider 基类"""

    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None,
                 timeout: float = 120.0, thinking: Optional[str] = None, **kwargs):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.thinking = thinking  # "enabled" / "disabled" / None (default)
        self.extra_params = kwargs

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stream: bool = True,
    ) -> CompletionResult:
        """发送补全请求并返回结果"""
        ...

    @abstractmethod
    async def stream_complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamChunk]:
        """流式补全"""
        ...

    async def close(self):
        """清理资源"""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model!r})"
