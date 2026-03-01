"""
OpenAI 兼容 Provider
支持: OpenAI, Kimi (Moonshot), MiniMax, DeepSeek, 以及任何 OpenAI 兼容 API
"""

from __future__ import annotations

import time
from typing import AsyncIterator, Optional

from openai import AsyncOpenAI

from .base import Provider, StreamChunk, CompletionResult


class OpenAICompatProvider(Provider):
    """OpenAI 兼容的 API Provider"""

    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None,
                 timeout: float = 120.0, **kwargs):
        super().__init__(model, api_key, base_url, timeout, **kwargs)
        import httpx
        from ._async_backend import AsyncIOBackend
        transport = httpx.AsyncHTTPTransport()
        transport._pool._network_backend = AsyncIOBackend()
        http_client = httpx.AsyncClient(transport=transport, timeout=timeout)
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
        )

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stream: bool = True,
    ) -> CompletionResult:
        result = CompletionResult(text="", model=self.model)
        result.request_start = time.time()

        try:
            if stream:
                full_text = []
                async for chunk in self.stream_complete(messages, temperature, max_tokens):
                    if not result.first_token_time and chunk.text:
                        result.first_token_time = chunk.timestamp
                    full_text.append(chunk.text)
                    result.chunk_timestamps.append(chunk.timestamp)
                    result.chunk_texts.append(chunk.text)

                result.text = "".join(full_text)
                result.end_time = time.time()
                # 估算 token 数
                result.completion_tokens = max(1, len(result.text) // 3)
            else:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                )
                result.first_token_time = time.time()
                result.end_time = result.first_token_time
                result.text = response.choices[0].message.content or ""
                if response.usage:
                    result.prompt_tokens = response.usage.prompt_tokens
                    result.completion_tokens = response.usage.completion_tokens

        except Exception as e:
            result.error = str(e)
            result.end_time = time.time()

        return result

    async def stream_complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamChunk]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield StreamChunk(
                    text=chunk.choices[0].delta.content,
                    timestamp=time.time(),
                    finish_reason=chunk.choices[0].finish_reason,
                )
            # 尝试获取 usage 信息 (stream 结束时)
            if hasattr(chunk, "usage") and chunk.usage:
                pass  # usage 在 CompletionResult 中处理

    async def close(self):
        await self.client.close()
