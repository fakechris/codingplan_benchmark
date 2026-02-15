# LLM Coding Benchmark

云端 LLM 编码能力评测工具 — 检测**量化、掺水、限速**的模型。

通过标准化的编码任务 (代码生成、Bug 修复、重构、算法、代码审查等) 对多个 LLM 进行横向对比, 输出 **客观性能指标** 和 **代码质量评分**。

## 评测维度

| 维度 | 权重 | 说明 |
|------|------|------|
| 质量 Quality | 45% | 规则评分 + LLM Judge (可选) |
| 速度 Speed | 25% | TTFT + TPS 综合 |
| 吞吐 Throughput | 15% | 并发 RPS + 成功率 |
| 一致性 Consistency | 15% | 多次运行的稳定性 (检测掺水) |

## 客观指标

每次评测自动采集以下 raw metrics:

- **TTFT** (Time to First Token) — 首字延迟, 从发送请求到收到第一个 token 的时间
- **TPS** (Tokens Per Second) — 每秒输出 token 数 (去除 TTFT 后的纯生成速度)
- **CPS** (Characters Per Second) — 每秒输出字符数
- **Gen Time** — 纯生成时间 (总延迟 - TTFT)
- **Total Latency** — 端到端总延迟
- **Output Tokens / Chars** — 输出量
- **Wall Time** — 整个模型完成全部测试的挂钟时间
- **RPS** — 并发吞吐 (请求/秒)
- **P50 / P95 / P99 延迟** — 吞吐测试的延迟分位数

所有指标提供 min / max / avg / p50 统计。

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 创建配置

```bash
cp config.example.yaml config.yaml
# 编辑 config.yaml, 填入你的 API Key
```

### 3. 连通性测试

```bash
python -m llm_bench debug -c config.yaml
```

### 4. 运行评测

```bash
# 完整评测
python -m llm_bench run -c config.yaml

# 只跑 easy 难度
python -m llm_bench run -c config.yaml -d easy

# 只测某个模型
python -m llm_bench run -c config.yaml -m MiniMax-M2.5

# 禁用 Judge (更快)
python -m llm_bench run -c config.yaml --no-judge
```

### 5. 其他命令

```bash
# 查看所有评测任务
python -m llm_bench tasks

# 查看支持的 Provider
python -m llm_bench providers

# 对比历史结果
python -m llm_bench compare results/a.json results/b.json

# 快速测试单个模型 (无需配置文件)
python -m llm_bench quick -p openai -m gpt-4o -k sk-xxx
```

---

## 配置说明

`config.yaml` 支持完整的 per-model 参数配置:

```yaml
settings:
  temperature: 0.0           # 全局温度
  concurrency: 2             # 吞吐并发数
  throughput_multiplier: 1   # 吞吐请求 = concurrency × multiplier
  consistency_runs: 1        # 一致性测试轮数
  max_quality_tasks: 2       # 质量题数上限 (0=不限)
  anti_cache: true           # 防缓存 nonce
  enable_judge: false        # LLM Judge 评分

models:
  - name: My-Model
    provider: openai_compat    # openai / anthropic / openai_compat / anthropic_compat
    model: model-id
    base_url: https://...
    api_key: "sk-xxx"
    max_tokens: 8192
    temperature: 0.0           # per-model 覆盖
    timeout: 120
    retry_count: 2
    # enabled: false           # 临时禁用
```

支持的 Provider: `openai`, `anthropic`, `openai_compat`, `anthropic_compat`, `kimi`, `minimax`, `deepseek`, `qwen`, `zhipu`, `doubao`, `baichuan`, `yi`, `xai`, `gemini`, `mistral`

### 测试量控制

| 参数 | 默认 | 精简 | 说明 |
|------|------|------|------|
| `max_quality_tasks` | 0 (不限) | 2 | 每次评测的题目数 |
| `throughput_multiplier` | 3 | 1 | 吞吐请求倍数 |
| `consistency_runs` | 3 | 1 | 一致性重复次数 |

---

## 评测任务

共 19 道编码任务, 涵盖:

| 类别 | 数量 | 难度分布 |
|------|------|----------|
| 代码生成 (Code Gen) | 6 | Easy ~ Hard |
| 算法 (Algorithm) | 3 | Easy ~ Hard |
| Bug 修复 (Bug Fix) | 3 | Easy ~ Medium |
| 重构 (Refactoring) | 2 | Medium |
| 测试编写 (Test Writing) | 2 | Medium |
| 代码审查 (Code Review) | 2 | Medium ~ Hard |
| 系统设计 (System Design) | 1 | Expert |

语言覆盖: Python, Go, TypeScript

---

## 评测报告: 2026-02-15

**测试环境**: macOS, Python 3.9.6, 网络环境 (中国大陆)  
**测试配置**: easy 难度, 2 道质量题, 吞吐并发 2, anti-cache 开启  
**测试时间**: 2026-02-15 17:45 ~ 17:59 (UTC+8)

### 被测模型

| 模型 | Provider | API 协议 | Endpoint | max_tokens |
|------|----------|----------|----------|------------|
| MiniMax-M2.5 | MiniMax | Anthropic 兼容 | api.minimax.io | 8192 |
| Kimi-K2.5 | 火山引擎 ARK | Anthropic 兼容 | ark.cn-beijing.volces.com | 16384 |
| Doubao-Seed-2.0-Code | 火山引擎 ARK | Anthropic 兼容 | ark.cn-beijing.volces.com | 8192 |

### 综合排行

| # | 模型 | 综合 | 质量 | 速度 | 吞吐 | 一致性 | 等级 |
|---|------|------|------|------|------|--------|------|
| 1 | **Doubao-Seed-2.0-Code** | **73.3** | 60.1 | **89.1** | **59.8** | 100.0 | **B** |
| 2 | MiniMax-M2.5 | 61.2 | **63.8** | 38.1 | 53.0 | 100.0 | C |
| 3 | Kimi-K2.5 | 52.6 | 58.6 | 14.8 | 50.4 | 100.0 | D |

### 客观指标对比

#### TTFT (首字延迟)

| 指标 | Doubao-Seed-2.0-Code | MiniMax-M2.5 | Kimi-K2.5 |
|------|---------------------|-------------|-----------|
| 平均 TTFT | **293 ms** | 20.62 s | 143.13 s (2.4 min) |
| 最小 TTFT | **248 ms** | 15.88 s | 111.83 s (1.9 min) |
| 最大 TTFT | **339 ms** | 25.36 s | 174.42 s (2.9 min) |
| P50 TTFT | **339 ms** | 25.36 s | 174.42 s |

#### TPS (Tokens Per Second)

| 指标 | Doubao-Seed-2.0-Code | MiniMax-M2.5 | Kimi-K2.5 |
|------|---------------------|-------------|-----------|
| 平均 TPS | **82.8** tok/s | 65.3 tok/s | 28.5 tok/s |
| 最小 TPS | **80.2** tok/s | 30.3 tok/s | 24.1 tok/s |
| 最大 TPS | 85.4 tok/s | **100.4** tok/s | 32.9 tok/s |

#### CPS (Characters Per Second)

| 指标 | Doubao-Seed-2.0-Code | MiniMax-M2.5 | Kimi-K2.5 |
|------|---------------------|-------------|-----------|
| 平均 CPS | **248.8** char/s | 196.0 char/s | 85.5 char/s |
| 最小 CPS | **240.8** char/s | 90.9 char/s | 72.3 char/s |
| 最大 CPS | 256.8 char/s | **301.2** char/s | 98.6 char/s |

#### 延迟 & 耗时

| 指标 | Doubao-Seed-2.0-Code | MiniMax-M2.5 | Kimi-K2.5 |
|------|---------------------|-------------|-----------|
| 平均生成时间 | **9.35 s** | 25.26 s | 25.42 s |
| 平均总延迟 | **9.64 s** | 45.88 s | 168.55 s (2.8 min) |
| 最大总延迟 | **17.52 s** | 64.05 s (1.1 min) | 190.91 s (3.2 min) |
| 测试总耗时 (Wall Time) | **51.76 s** | 129.17 s (2.2 min) | 620.30 s (10.3 min) |

#### 输出量

| 指标 | Doubao-Seed-2.0-Code | MiniMax-M2.5 | Kimi-K2.5 |
|------|---------------------|-------------|-----------|
| 总输出 Tokens | 1,508 | 1,694 | 1,526 |
| 总输出字符 | 4,526 | 5,084 | 4,580 |

### 吞吐测试 (并发=2)

| 指标 | Doubao-Seed-2.0-Code | MiniMax-M2.5 | Kimi-K2.5 |
|------|---------------------|-------------|-----------|
| 成功率 | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) |
| RPS | **0.98** | 0.30 | 0.04 |
| 平均延迟 | **2.01 s** | 6.09 s | 41.44 s |
| P50 延迟 | **2.01 s** | 6.09 s | 41.44 s |
| P95 延迟 | **2.04 s** | 6.62 s | 52.12 s |
| 平均 TPS | 66.3 tok/s | **72.2** tok/s | 19.3 tok/s |

### 逐题详情

#### E01: IPv4 地址验证 (code_gen, easy)

| 指标 | Doubao-Seed-2.0-Code | MiniMax-M2.5 | Kimi-K2.5 |
|------|---------------------|-------------|-----------|
| TTFT | **0.339 s** | 25.357 s | 174.419 s |
| 生成时间 | **1.429 s** | 2.341 s | 16.487 s |
| 总延迟 | **1.768 s** | 27.698 s | 190.906 s |
| TPS | 85.4 tok/s | **100.4** tok/s | 24.1 tok/s |
| CPS | 256.8 char/s | **301.2** char/s | 72.3 char/s |
| 输出 Tokens | 122 | 235 | 397 |
| 输出字符 | 367 | 705 | 1,192 |
| 质量分 | 47.1 | **67.1** | 65.6 |

#### E02: LRU Cache 实现 (algorithm, easy)

| 指标 | Doubao-Seed-2.0-Code | MiniMax-M2.5 | Kimi-K2.5 |
|------|---------------------|-------------|-----------|
| TTFT | **0.248 s** | 15.882 s | 111.833 s |
| 生成时间 | **17.274 s** | 48.172 s | 34.357 s |
| 总延迟 | **17.522 s** | 64.054 s | 146.190 s |
| TPS | **80.2** tok/s | 30.3 tok/s | 32.9 tok/s |
| CPS | **240.8** char/s | 90.9 char/s | 98.6 char/s |
| 输出 Tokens | 1,386 | 1,459 | 1,129 |
| 输出字符 | 4,159 | 4,379 | 3,388 |
| 质量分 | **73.1** | 60.5 | 51.5 |

### 一致性测试 (量化/掺水检测)

| 任务 | 模型 | 平均分 | 标准差 | 相似度 | 判定 |
|------|------|--------|--------|--------|------|
| C01 | Doubao-Seed-2.0-Code | 42.2 | 0.00 | 1.00 | 稳定 |
| C01 | MiniMax-M2.5 | 42.9 | 0.00 | 1.00 | 稳定 |
| C01 | Kimi-K2.5 | 37.2 | 0.00 | 1.00 | 稳定 |

> 注: consistency_runs=1 时标准差为 0, 需调高轮数才能有效检测掺水行为。

### 分析

1. **Doubao-Seed-2.0-Code 速度碾压**: TTFT 仅 293ms (亚秒级), 比 MiniMax 快 **70 倍**, 比 Kimi 快 **488 倍**。吞吐 RPS 也是最高的 (0.98 vs 0.30 vs 0.04)。
2. **Kimi-K2.5 的 TTFT 异常高** (平均 2.4 分钟): 很可能是内部 reasoning/thinking 阶段在消耗大量时间后才开始输出 token。纯生成阶段 (Gen Time ≈ 25s) 和 MiniMax 差不多, 说明瓶颈不在网络或生成速度, 而在推理前的 "思考" 阶段。
3. **MiniMax 质量最高**: 尽管速度不如 Doubao, 但代码质量评分是三者中最好的 (63.8 vs 60.1 vs 58.6)。峰值 TPS 也达到 100.4 tok/s。
4. **三者一致性均为"稳定"**: 本次测试中未检测到明显的量化掺水行为, 但 consistency_runs=1 检测力度有限。

---

## 项目结构

```
benchmark/
├── llm_bench/
│   ├── __init__.py
│   ├── __main__.py          # CLI 入口
│   ├── config.py            # 配置管理
│   ├── runner.py            # 评测执行引擎
│   ├── scorer.py            # 规则评分 + Judge 评分
│   ├── report.py            # 报告生成 (终端 + JSON)
│   ├── providers/
│   │   ├── base.py          # Provider 抽象基类
│   │   ├── openai_compat.py # OpenAI 兼容 Provider
│   │   ├── anthropic_compat.py  # Anthropic 兼容 Provider
│   │   └── registry.py      # Provider 预设注册表
│   └── tasks/
│       └── coding_plans.py  # 评测任务集 (19 道)
├── config.example.yaml      # 配置模板 (无 API Key)
├── pyproject.toml
├── requirements.txt
└── results/                 # 评测结果输出
    └── benchmark_results.json
```

## License

MIT
