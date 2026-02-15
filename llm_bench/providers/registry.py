"""
Provider 注册表和预设配置
支持通过名称快速创建 provider, 也支持完全自定义
"""

from __future__ import annotations

from typing import Optional
from .base import Provider
from .openai_compat import OpenAICompatProvider
from .anthropic_compat import AnthropicCompatProvider


# 预设的 Provider 配置
PROVIDER_PRESETS: dict[str, dict] = {
    # === OpenAI ===
    "openai": {
        "class": "openai",
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
    },
    # === Anthropic ===
    "anthropic": {
        "class": "anthropic",
        "env_key": "ANTHROPIC_API_KEY",
    },
    # === Kimi (Moonshot AI) ===
    "kimi": {
        "class": "openai",
        "base_url": "https://api.moonshot.cn/v1",
        "env_key": "MOONSHOT_API_KEY",
    },
    # === MiniMax ===
    "minimax": {
        "class": "openai",
        "base_url": "https://api.minimax.chat/v1",
        "env_key": "MINIMAX_API_KEY",
    },
    # === DeepSeek ===
    "deepseek": {
        "class": "openai",
        "base_url": "https://api.deepseek.com/v1",
        "env_key": "DEEPSEEK_API_KEY",
    },
    # === 阿里通义千问 ===
    "qwen": {
        "class": "openai",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_key": "DASHSCOPE_API_KEY",
    },
    # === 智谱 GLM ===
    "zhipu": {
        "class": "openai",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "env_key": "ZHIPU_API_KEY",
    },
    # === 字节豆包 ===
    "doubao": {
        "class": "openai",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "env_key": "DOUBAO_API_KEY",
    },
    # === 百川 ===
    "baichuan": {
        "class": "openai",
        "base_url": "https://api.baichuan-ai.com/v1",
        "env_key": "BAICHUAN_API_KEY",
    },
    # === 零一万物 ===
    "yi": {
        "class": "openai",
        "base_url": "https://api.lingyiwanwu.com/v1",
        "env_key": "YI_API_KEY",
    },
    # === xAI Grok ===
    "grok": {
        "class": "openai",
        "base_url": "https://api.x.ai/v1",
        "env_key": "XAI_API_KEY",
    },
    # === Google Gemini (OpenAI compat) ===
    "gemini": {
        "class": "openai",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "env_key": "GOOGLE_API_KEY",
    },
    # === Mistral ===
    "mistral": {
        "class": "openai",
        "base_url": "https://api.mistral.ai/v1",
        "env_key": "MISTRAL_API_KEY",
    },
}


def get_provider(
    provider_name: str,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 120.0,
    **kwargs,
) -> Provider:
    """
    根据 provider 名称创建 Provider 实例

    Args:
        provider_name: 预设名称 (如 "openai", "kimi") 或类型 ("openai_compat", "anthropic_compat")
        model: 模型名称
        api_key: API Key (如果不提供, 从环境变量获取)
        base_url: 自定义 base URL (覆盖预设)
        timeout: 请求超时
    """
    import os

    # 查找预设
    preset = PROVIDER_PRESETS.get(provider_name, {})
    provider_class_name = preset.get("class", provider_name)

    # 确定 API key
    if not api_key:
        env_key = preset.get("env_key", f"{provider_name.upper()}_API_KEY")
        api_key = os.environ.get(env_key, "")
        if not api_key:
            raise ValueError(
                f"未找到 API key. 请设置环境变量 {env_key} 或在配置中指定 api_key"
            )

    # 确定 base URL
    if not base_url:
        base_url = preset.get("base_url")

    # 创建 provider
    if provider_class_name in ("openai", "openai_compat"):
        return OpenAICompatProvider(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            **kwargs,
        )
    elif provider_class_name in ("anthropic", "anthropic_compat"):
        return AnthropicCompatProvider(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            **kwargs,
        )
    else:
        raise ValueError(
            f"未知的 provider: {provider_name}. "
            f"可用预设: {', '.join(PROVIDER_PRESETS.keys())}. "
            f"或使用 'openai_compat' / 'anthropic_compat'"
        )
