"""
配置管理 - 支持 YAML 配置文件

每个模型支持独立的全部参数配置，类似 Claude Code 的配置风格:
  - provider / model / api_key / base_url 等连接参数
  - temperature / max_tokens / top_p 等生成参数
  - system_prompt / thinking_budget / custom_headers 等高级参数
  - timeout / retry 等可靠性参数
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

import yaml


@dataclass
class ModelConfig:
    """单个模型的完整配置"""
    # === 基本信息 ===
    name: str                             # 显示名称 (如 "Claude-3.5-Sonnet")
    provider: str                         # provider 类型 (openai / anthropic / kimi / minimax / ...)
    model: str                            # 模型 ID (如 gpt-4o, claude-3-5-sonnet-20241022)

    # === 连接参数 ===
    api_key: Optional[str] = None         # API Key (优先级: 直接指定 > env_key > 预设环境变量)
    env_key: Optional[str] = None         # 环境变量名 (如 OPENAI_API_KEY)
    base_url: Optional[str] = None        # 自定义 API 地址
    timeout: float = 120.0                # 请求超时 (秒)
    custom_headers: dict = field(default_factory=dict)  # 自定义 HTTP headers

    # === 生成参数 ===
    temperature: Optional[float] = None   # 生成温度 (None = 用全局默认)
    max_tokens: Optional[int] = None      # 最大输出 token (None = 用任务默认)
    top_p: Optional[float] = None         # nucleus sampling
    top_k: Optional[int] = None           # top-k sampling
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[list[str]] = None      # 停止词

    # === 高级参数 ===
    system_prompt_override: Optional[str] = None  # 覆盖任务的 system prompt
    thinking_budget: Optional[int] = None         # Claude extended thinking token budget
    stream: bool = True                           # 是否使用流式输出
    seed: Optional[int] = None                    # 随机种子 (可复现)

    # === 可靠性 ===
    retry_count: int = 2                  # 失败重试次数
    retry_delay: float = 1.0              # 重试间隔 (秒)

    # === 过滤 ===
    enabled: bool = True                  # 是否启用该模型

    # === 扩展字段 (provider 特定) ===
    extra: dict = field(default_factory=dict)

    def get_api_key(self) -> str:
        """按优先级获取 API key"""
        if self.api_key:
            return self.api_key
        if self.env_key:
            key = os.environ.get(self.env_key, "")
            if key:
                return key
        # 从 provider 预设的环境变量获取
        from .providers.registry import PROVIDER_PRESETS
        preset = PROVIDER_PRESETS.get(self.provider, {})
        default_env = preset.get("env_key", f"{self.provider.upper()}_API_KEY")
        return os.environ.get(default_env, "")

    def get_temperature(self, global_temp: float) -> float:
        """获取温度 (per-model 优先, 否则用全局)"""
        return self.temperature if self.temperature is not None else global_temp

    def get_max_tokens(self, task_default: int) -> int:
        """获取 max_tokens (per-model 优先, 否则用任务默认)"""
        return self.max_tokens if self.max_tokens is not None else task_default


@dataclass
class BenchmarkConfig:
    """Benchmark 全局配置"""
    models: list[ModelConfig] = field(default_factory=list)
    judge_model: Optional[ModelConfig] = None

    # === 全局默认 (可被 per-model 覆盖) ===
    temperature: float = 0.0
    concurrency: int = 5
    consistency_runs: int = 3
    enable_judge: bool = True
    parallel_models: bool = False         # 是否并行测试多个模型

    # === 测试量控制 ===
    throughput_multiplier: int = 3        # 吞吐请求数 = concurrency * multiplier
    max_quality_tasks: int = 0            # 质量题数上限 (0=不限)
    anti_cache: bool = False              # 给 prompt 加随机 nonce 防缓存

    # === 任务过滤 ===
    task_ids: Optional[list[str]] = None
    difficulty: Optional[str] = None

    # === 输出 ===
    output_dir: str = "./results"
    output_format: list[str] = field(default_factory=lambda: ["terminal", "json"])


def _parse_model_config(raw: dict) -> ModelConfig:
    """从字典解析单个模型配置"""
    return ModelConfig(
        name=raw.get("name", raw.get("model", "unknown")),
        provider=raw["provider"],
        model=raw["model"],
        api_key=raw.get("api_key"),
        env_key=raw.get("env_key"),
        base_url=raw.get("base_url"),
        timeout=raw.get("timeout", 120.0),
        custom_headers=raw.get("custom_headers", {}),
        temperature=raw.get("temperature"),
        max_tokens=raw.get("max_tokens"),
        top_p=raw.get("top_p"),
        top_k=raw.get("top_k"),
        frequency_penalty=raw.get("frequency_penalty"),
        presence_penalty=raw.get("presence_penalty"),
        stop=raw.get("stop"),
        system_prompt_override=raw.get("system_prompt"),
        thinking_budget=raw.get("thinking_budget"),
        stream=raw.get("stream", True),
        seed=raw.get("seed"),
        retry_count=raw.get("retry_count", 2),
        retry_delay=raw.get("retry_delay", 1.0),
        enabled=raw.get("enabled", True),
        extra=raw.get("extra", {}),
    )


def load_config(config_path: str) -> BenchmarkConfig:
    """从 YAML 文件加载配置"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    config = BenchmarkConfig()

    # 全局设置
    settings = raw.get("settings", {})
    config.temperature = settings.get("temperature", 0.0)
    config.concurrency = settings.get("concurrency", 5)
    config.consistency_runs = settings.get("consistency_runs", 3)
    config.enable_judge = settings.get("enable_judge", True)
    config.output_dir = settings.get("output_dir", "./results")
    config.output_format = settings.get("output_format", ["terminal", "json"])
    config.parallel_models = settings.get("parallel_models", False)
    config.throughput_multiplier = settings.get("throughput_multiplier", 3)
    config.max_quality_tasks = settings.get("max_quality_tasks", 0)
    config.anti_cache = settings.get("anti_cache", False)

    if "task_ids" in settings:
        config.task_ids = settings["task_ids"]
    if "difficulty" in settings:
        config.difficulty = settings["difficulty"]

    # 模型配置
    for m in raw.get("models", []):
        mc = _parse_model_config(m)
        if mc.enabled:
            config.models.append(mc)

    # Judge 模型
    judge = raw.get("judge")
    if judge:
        config.judge_model = _parse_model_config(judge)

    return config


def create_default_config() -> str:
    """生成默认配置文件内容"""
    return '''# LLM Coding Benchmark 配置文件
# ====================================
# 每个模型都支持完整的独立参数配置

settings:
  temperature: 0.0          # 全局默认温度 (各模型可独立覆盖)
  concurrency: 5            # 吞吐测试并发数
  consistency_runs: 3       # 一致性测试重复次数
  enable_judge: true        # 是否启用 Judge 模型评分
  parallel_models: false    # 是否并行测试多个模型 (true 更快但日志交错)
  output_dir: ./results
  output_format:
    - terminal
    - json
  # task_ids:               # 指定任务 ID (不指定则运行全部)
  #   - E01
  #   - M01
  #   - H01
  # difficulty: medium      # 过滤难度: easy / medium / hard / expert

# Judge 模型 (用于质量评分, 建议用强模型)
judge:
  provider: openai
  model: gpt-4o
  temperature: 0.0
  max_tokens: 1024
  # api_key: sk-xxx         # 或设置环境变量 OPENAI_API_KEY

# ====================================
# 待测试的模型列表
# ====================================
# 每个模型支持以下所有参数 (都是可选的, 有合理默认值):
#
# 连接参数:
#   provider:         必填, 类型名 (openai/anthropic/kimi/minimax/deepseek/qwen/zhipu/doubao/...)
#   model:            必填, 模型 ID
#   name:             显示名称 (默认=model)
#   api_key:          API Key (直接填写)
#   env_key:          从指定环境变量读取 API Key
#   base_url:         自定义 API 地址
#   timeout:          请求超时秒数 (默认 120)
#   custom_headers:   自定义 HTTP 请求头
#
# 生成参数:
#   temperature:      温度 (不填则用全局 settings.temperature)
#   max_tokens:       最大输出 token (不填则用任务默认)
#   top_p:            nucleus sampling
#   top_k:            top-k sampling
#   frequency_penalty: 频率惩罚
#   presence_penalty:  存在惩罚
#   stop:             停止词列表
#   seed:             随机种子 (可复现)
#
# 高级参数:
#   system_prompt:    覆盖任务默认的 system prompt
#   thinking_budget:  Claude extended thinking token 预算
#   stream:           是否流式输出 (默认 true, 用于测 TTFT/TPS)
#
# 可靠性:
#   retry_count:      失败重试次数 (默认 2)
#   retry_delay:      重试间隔秒数 (默认 1.0)
#
# 控制:
#   enabled:          是否启用 (默认 true, 设 false 可临时禁用)
#
# 扩展:
#   extra:            provider 特定的额外参数 (dict)

models:
  # === OpenAI ===
  - name: GPT-4o
    provider: openai
    model: gpt-4o
    # api_key: sk-xxx
    # temperature: 0.0
    # max_tokens: 4096

  # === Anthropic Claude ===
  - name: Claude-3.5-Sonnet
    provider: anthropic
    model: claude-3-5-sonnet-20241022
    # api_key: sk-ant-xxx
    # thinking_budget: 10000   # extended thinking

  # === Kimi (Moonshot) ===
  - name: Kimi
    provider: kimi
    model: moonshot-v1-128k
    # env_key: MOONSHOT_API_KEY
    # temperature: 0.3

  # === MiniMax ===
  - name: MiniMax-Text-01
    provider: minimax
    model: MiniMax-Text-01
    # env_key: MINIMAX_API_KEY

  # === DeepSeek ===
  - name: DeepSeek-V3
    provider: deepseek
    model: deepseek-chat
    # env_key: DEEPSEEK_API_KEY

  # === 通义千问 ===
  - name: Qwen-Max
    provider: qwen
    model: qwen-max
    # env_key: DASHSCOPE_API_KEY

  # === 智谱 GLM ===
  - name: GLM-4-Plus
    provider: zhipu
    model: glm-4-plus
    # env_key: ZHIPU_API_KEY

  # === 自定义 OpenAI 兼容服务 ===
  # - name: My-Local-Model
  #   provider: openai_compat
  #   model: my-model-v1
  #   base_url: http://localhost:8080/v1
  #   api_key: dummy-key
  #   temperature: 0.7
  #   max_tokens: 2048
  #   timeout: 60
  #   stream: true
  #   custom_headers:
  #     X-Custom-Header: my-value
'''
