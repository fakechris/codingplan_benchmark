"""
Anthropic 兼容 Provider
支持: Claude (Opus, Sonnet, Haiku) 以及兼容 Anthropic API 的服务
"""

from __future__ import annotations

import time
from typing import AsyncIterator, Optional

from anthropic import AsyncAnthropic

from .base import Provider, StreamChunk, CompletionResult


class AnthropicCompatProvider(Provider):
    """Anthropic API Provider"""

    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None,
                 timeout: float = 120.0, **kwargs):
        super().__init__(model, api_key, base_url, timeout, **kwargs)
        client_kwargs = {
            "api_key": api_key,
            "timeout": timeout,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = AsyncAnthropic(**client_kwargs)

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stream: bool = True,
    ) -> CompletionResult:
        result = CompletionResult(text="", model=self.model)
        result.request_start = time.time()

        # 从 messages 中提取 system 消息
        system_msg = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                chat_messages.append(msg)

        try:
            if stream:
                full_text = []
                async for chunk in self._stream_impl(chat_messages, system_msg, temperature, max_tokens):
                    if not result.first_token_time and chunk.text:
                        result.first_token_time = chunk.timestamp
                    full_text.append(chunk.text)
                    result.chunk_timestamps.append(chunk.timestamp)
                    result.chunk_texts.append(chunk.text)

                result.text = "".join(full_text)
                result.end_time = time.time()
                result.completion_tokens = max(1, len(result.text) // 3)
            else:
                kwargs = {
                    "model": self.model,
                    "messages": chat_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if system_msg:
                    kwargs["system"] = system_msg

                response = await self.client.messages.create(**kwargs)
                result.first_token_time = time.time()
                result.end_time = result.first_token_time
                result.text = response.content[0].text if response.content else ""
                result.prompt_tokens = response.usage.input_tokens
                result.completion_tokens = response.usage.output_tokens

        except Exception as e:
            result.error = str(e)
            result.end_time = time.time()

        return result

    async def _stream_impl(
        self,
        messages: list[dict],
        system: str,
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[StreamChunk]:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system:
            kwargs["system"] = system

        async with self.client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield StreamChunk(
                    text=text,
                    timestamp=time.time(),
                )

    async def stream_complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamChunk]:
        system_msg = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                chat_messages.append(msg)

        async for chunk in self._stream_impl(chat_messages, system_msg, temperature, max_tokens):
            yield chunk

    async def close(self):
        await self.client.close()
