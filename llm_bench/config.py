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
    thinking: Optional[str] = None                # Thinking 模式: "enabled" / "disabled" / None(用模型默认)
    stream: bool = True                           # 是否使用流式输出
    seed: Optional[int] = None                    # 随机种子 (可复现)

    # === 可靠性 ===
    retry_count: int = 2                  # 失败重试次数
    retry_delay: float = 1.0              # 重试间隔 (秒)

    # === 过滤 ===
    enabled: bool = True                  # 是否启用该模型

    # === 来源 ===
    coding_plan: Optional[str] = None     # 所属 Coding Plan ID (两层配置时自动填充)

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
        thinking=raw.get("thinking"),
        stream=raw.get("stream", True),
        seed=raw.get("seed"),
        retry_count=raw.get("retry_count", 2),
        retry_delay=raw.get("retry_delay", 1.0),
        enabled=raw.get("enabled", True),
        extra=raw.get("extra", {}),
    )


def _parse_coding_plans(raw_plans: dict) -> list[ModelConfig]:
    """解析两层结构的 coding_plans 配置，展平为 ModelConfig 列表

    两层结构:
      coding_plans:
        plan_id:
          name: "显示名称"
          api_key: "共享 Key"
          api_type: anthropic / openai    # 默认 API 协议
          base_url: "https://..."         # 默认 Base URL
          timeout: 120                    # 默认超时
          max_tokens: 32768               # 默认 max_tokens
          retry_count: 2                  # 默认重试
          models:
            - name: "Model-A"
              model: "model-id"
              thinking: enabled           # 可选
              api_type: openai            # 可选, 覆盖 plan 级别
              base_url: "https://..."     # 可选, 覆盖 plan 级别
    """
    _api_type_to_provider = {
        "anthropic": "anthropic_compat",
        "openai": "openai_compat",
    }

    models: list[ModelConfig] = []
    for plan_id, plan_data in raw_plans.items():
        if not isinstance(plan_data, dict):
            continue

        # Plan 级别默认值
        plan_api_key = plan_data.get("api_key")
        plan_env_key = plan_data.get("env_key")
        plan_api_type = plan_data.get("api_type", "anthropic")
        plan_base_url = plan_data.get("base_url")
        plan_retry = plan_data.get("retry_count", 2)
        plan_retry_delay = plan_data.get("retry_delay", 1.0)
        plan_timeout = plan_data.get("timeout", 120.0)
        plan_max_tokens = plan_data.get("max_tokens")

        for m in plan_data.get("models", []):
            # Model 级别可覆盖 Plan 级别的 api_type / base_url
            model_api_type = m.get("api_type", plan_api_type)
            provider = _api_type_to_provider.get(model_api_type, model_api_type)

            mc = ModelConfig(
                name=m.get("name", m.get("model", "unknown")),
                provider=provider,
                model=m["model"],
                api_key=m.get("api_key", plan_api_key),
                env_key=m.get("env_key", plan_env_key),
                base_url=m.get("base_url", plan_base_url),
                timeout=m.get("timeout", plan_timeout),
                max_tokens=m.get("max_tokens", plan_max_tokens),
                retry_count=m.get("retry_count", plan_retry),
                retry_delay=m.get("retry_delay", plan_retry_delay),
                thinking=m.get("thinking"),
                stream=m.get("stream", True),
                enabled=m.get("enabled", True),
                temperature=m.get("temperature"),
                custom_headers=m.get("custom_headers", {}),
                extra=m.get("extra", {}),
                coding_plan=plan_id,
            )
            if mc.enabled:
                models.append(mc)

    return models


def load_config(config_path: str) -> BenchmarkConfig:
    """从 YAML 文件加载配置

    支持两种模型配置格式 (可同时使用):
      1. coding_plans: 两层结构 (推荐, Plan 级别共享 key/url)
      2. models: 平铺结构 (向后兼容, 每个模型独立配置)
    """
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

    # 两层结构: coding_plans (推荐)
    coding_plans = raw.get("coding_plans", {})
    if coding_plans:
        config.models.extend(_parse_coding_plans(coding_plans))

    # 平铺结构: models (向后兼容)
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
# 采用两层结构: Coding Plan → Models

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
# judge:
#   provider: openai
#   model: gpt-4o
#   api_key: "sk-xxx"

# ====================================
# Coding Plans - 两层配置结构
# ====================================
#
# 第一层 - Coding Plan:
#   api_key:        共享 API Key (该 Plan 下所有模型共用)
#   env_key:        或从环境变量读取 API Key
#   api_type:       API 协议类型 (anthropic / openai), 默认 anthropic
#   base_url:       默认 API 地址
#   timeout:        默认超时秒数 (默认 120)
#   max_tokens:     默认最大输出 token
#   retry_count:    默认重试次数 (默认 2)
#
# 第二层 - Model (可覆盖 Plan 级别的任何字段):
#   model:          必填, 模型 ID
#   name:           显示名称 (默认=model)
#   api_type:       覆盖 Plan 的 API 协议
#   base_url:       覆盖 Plan 的 API 地址
#   thinking:       Thinking 模式: enabled / disabled
#   timeout:        覆盖超时
#   max_tokens:     覆盖 max_tokens
#   temperature:    覆盖温度
#   enabled:        是否启用 (默认 true)

coding_plans:
  # === 豆包 Coding Plan (火山引擎 ARK, Anthropic 兼容) ===
  doubao:
    name: "豆包"
    api_type: anthropic
    base_url: https://ark.cn-beijing.volces.com/api/coding
    api_key: "<YOUR_VOLCENGINE_API_KEY>"
    max_tokens: 32768
    models:
      - name: Doubao-Default
        model: doubao-seed-2.0-code
        timeout: 120
      - name: Doubao-Think-On
        model: doubao-seed-2.0-code
        timeout: 300
        thinking: enabled
      - name: Kimi-Think-On
        model: kimi-k2.5
        timeout: 300
        thinking: enabled
      - name: Kimi-Think-Off
        model: kimi-k2.5
        timeout: 120
        thinking: disabled

  # === Kimi Coding Plan (月之暗面, Anthropic 兼容) ===
  kimi:
    name: "Kimi"
    api_type: anthropic
    base_url: https://api.kimi.com/coding
    api_key: "<YOUR_KIMI_API_KEY>"
    max_tokens: 32768
    models:
      - name: Kimi-Native-Think-On
        model: kimi-k2.5
        timeout: 300
        thinking: enabled
      - name: Kimi-Native-Think-Off
        model: kimi-k2.5
        timeout: 120
        thinking: disabled

  # === MiniMax Coding Plan (Anthropic 兼容) ===
  minimax:
    name: "MiniMax"
    api_type: anthropic
    base_url: https://api.minimaxi.com/anthropic
    api_key: "<YOUR_MINIMAX_API_KEY>"
    max_tokens: 16384
    timeout: 180
    models:
      - name: MiniMax-M2.5
        model: MiniMax-M2.5

  # === 阿里云百炼 Coding Plan ===
  ali:
    name: "阿里云百炼"
    api_key: "<YOUR_DASHSCOPE_API_KEY>"
    api_type: openai
    base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
    max_tokens: 32768
    timeout: 180
    models:
      - name: Qwen3-Coder-Plus
        model: qwen3-coder-plus
      - name: Qwen3.5-Plus
        model: qwen3.5-plus
      # Anthropic 兼容模型 (覆盖 api_type 和 base_url)
      # - name: Ali-Kimi-Think-On
      #   model: kimi-k2.5
      #   api_type: anthropic
      #   base_url: https://dashscope.aliyuncs.com/apps/anthropic
      #   timeout: 300
      #   thinking: enabled

  # === 联通云 CUCloud Coding Plan (OpenAI 兼容) ===
  # cucloud:
  #   name: "联通云 CUCloud"
  #   api_type: openai
  #   base_url: https://aigw-gzgy2.cucloud.cn:8443/v1
  #   api_key: "<YOUR_CUCLOUD_API_KEY>"
  #   max_tokens: 16384
  #   timeout: 600
  #   models:
  #     - name: CUCloud-GLM-5
  #       model: glm-5

# ====================================
# 平铺模型列表 (向后兼容, 可与 coding_plans 同时使用)
# ====================================
# models:
#   - name: GPT-4o
#     provider: openai
#     model: gpt-4o
#     api_key: "sk-xxx"
'''
