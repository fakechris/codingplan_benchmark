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
                thinking_text = []
                in_thinking = False

                async for chunk in self._stream_events(chat_messages, system_msg, temperature, max_tokens):
                    now = time.time()

                    if chunk.get("type") == "thinking_start":
                        in_thinking = True
                        if not result.thinking_start_time:
                            result.thinking_start_time = now
                        continue

                    if chunk.get("type") == "thinking_end":
                        in_thinking = False
                        result.thinking_end_time = now
                        continue

                    if chunk.get("type") == "thinking_delta":
                        thinking_text.append(chunk.get("text", ""))
                        continue

                    # 正常 text delta
                    text = chunk.get("text", "")
                    if text:
                        if not result.first_token_time:
                            result.first_token_time = now
                        full_text.append(text)
                        result.chunk_timestamps.append(now)
                        result.chunk_texts.append(text)

                    # usage 事件 (message_delta with usage)
                    if "output_tokens" in chunk:
                        result.completion_tokens = chunk["output_tokens"]
                    if "input_tokens" in chunk:
                        result.prompt_tokens = chunk["input_tokens"]

                result.text = "".join(full_text)
                result.thinking_text = "".join(thinking_text)
                result.thinking_tokens = max(1, len(result.thinking_text) // 3) if result.thinking_text else 0
                result.end_time = time.time()
                # 如果 API 没返回 token 数, 用字符估算
                if result.completion_tokens == 0:
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

                # 解析 content blocks — 区分 thinking 和 text
                for block in (response.content or []):
                    if getattr(block, "type", "") == "thinking":
                        result.thinking_text = getattr(block, "thinking", "")
                        result.thinking_tokens = max(1, len(result.thinking_text) // 3)
                    elif getattr(block, "type", "") == "text":
                        result.text = block.text

                if not result.text and response.content:
                    result.text = response.content[0].text if hasattr(response.content[0], 'text') else ""

                result.prompt_tokens = response.usage.input_tokens
                result.completion_tokens = response.usage.output_tokens

        except Exception as e:
            result.error = str(e)
            result.end_time = time.time()

        return result

    async def _stream_events(
        self,
        messages: list[dict],
        system: str,
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[dict]:
        """底层 event stream — 区分 thinking / text / usage 事件"""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system:
            kwargs["system"] = system

        async with self.client.messages.stream(**kwargs) as stream:
            async for event in stream:
                etype = getattr(event, "type", "")

                # content_block_start: 识别 thinking vs text block
                if etype == "content_block_start":
                    cb = getattr(event, "content_block", None)
                    if cb and getattr(cb, "type", "") == "thinking":
                        yield {"type": "thinking_start"}
                    continue

                # content_block_stop
                if etype == "content_block_stop":
                    # 通过已有状态判断是否 thinking 结束
                    yield {"type": "thinking_end"}
                    continue

                # content_block_delta
                if etype == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if delta:
                        delta_type = getattr(delta, "type", "")
                        if delta_type == "thinking_delta":
                            yield {"type": "thinking_delta", "text": getattr(delta, "thinking", "")}
                        elif delta_type == "text_delta":
                            yield {"type": "text_delta", "text": getattr(delta, "text", "")}
                    continue

                # message_delta (final usage)
                if etype == "message_delta":
                    usage = getattr(event, "usage", None)
                    if usage:
                        yield {"output_tokens": getattr(usage, "output_tokens", 0)}
                    continue

                # message_start (input usage)
                if etype == "message_start":
                    msg = getattr(event, "message", None)
                    if msg:
                        usage = getattr(msg, "usage", None)
                        if usage:
                            yield {"input_tokens": getattr(usage, "input_tokens", 0)}
                    continue

    async def _stream_impl(
        self,
        messages: list[dict],
        system: str,
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[StreamChunk]:
        """兼容旧接口 — 只 yield text chunks"""
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
