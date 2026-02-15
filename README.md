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
- **Thinking Time** — 推理模型的"思考"阶段耗时 (如 Kimi K2.5 reasoning 模式)
- **Thinking Tokens** — 思考阶段消耗的 token 数

所有指标提供 min / max / avg / p50 统计。对于支持 reasoning/thinking 的模型, TTFT 会精确排除思考阶段, 仅反映首个文本 token 的延迟。

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

## 评测报告: 2026-02-15 (完整版)

**测试环境**: macOS, Python 3.9.6, 网络环境 (中国大陆)  
**测试配置**: 9 道 medium/hard/expert 实战题, 吞吐并发 2×2, 一致性 2 轮, anti-cache 开启  
**测试时间**: 2026-02-15 19:05 ~ 20:10 (UTC+8)  
**测试顺序**: Kimi → Doubao → MiniMax

### 被测模型

| 模型 | Provider | API 协议 | Endpoint | max_tokens |
|------|----------|----------|----------|------------|
| Kimi-K2.5 | 火山引擎 ARK | Anthropic 兼容 | ark.cn-beijing.volces.com | 16384 |
| Doubao-Seed-2.0-Code | 火山引擎 ARK | Anthropic 兼容 | ark.cn-beijing.volces.com | 8192 |
| MiniMax-M2.5 | MiniMax | Anthropic 兼容 | api.minimax.io | 8192 |

### 测试任务 (9 道)

| ID | 任务 | 难度 | 类别 |
|----|------|------|------|
| M01 | 并发安全的连接池 (asyncio) | Medium | 代码生成 |
| M03 | 修复 Go Race Condition | Medium | Bug 修复 |
| M04 | 代码审查: Node.js API 安全 | Medium | 代码审查 |
| M05 | 编写 pytest 单元测试 | Medium | 测试编写 |
| H01 | 实现跳表 (Skip List) | Hard | 算法 |
| H03 | 重构: 消除 God Object (SOLID) | Hard | 重构 |
| X01 | Go 并发 Bug 修复 | Expert | Bug 修复 |
| X02 | Promise.allSettled + 重试 | Expert | 代码生成 |
| S02 | 长代码生成: REST 框架 | Medium | 代码生成 |

### 综合排行

| # | 模型 | 综合 | 质量 | 速度 | 吞吐 | 一致性 | 等级 | 总耗时 |
|---|------|------|------|------|------|--------|------|--------|
| 1 | **Doubao-Seed-2.0-Code** | **80.2** | 78.1 | **91.7** | **60.0** | 87.5 | **A** | **6.1 min** |
| 2 | MiniMax-M2.5 | 68.8 | **80.0** | 39.2 | 53.7 | **100.0** | C | 16.1 min |
| 3 | Kimi-K2.5 | 63.4 | **80.0** | 19.3 | 50.6 | **100.0** | C | 42.8 min |

### 客观指标对比

#### TTFT (首字延迟)

| 指标 | Doubao-Seed-2.0-Code | MiniMax-M2.5 | Kimi-K2.5 |
|------|---------------------|-------------|-----------|
| 平均 TTFT | **317 ms** | 35.71 s | 2.1 min |
| 最小 TTFT | **234 ms** | 11.99 s | 19.51 s |
| 最大 TTFT | **465 ms** | 1.2 min | 4.6 min |
| P50 TTFT | **294 ms** | 28.49 s | 1.8 min |

> Kimi-K2.5 的 TTFT 包含内部 reasoning/thinking 阶段 (推测占 TTFT 的 90%+)。

#### TPS (Tokens Per Second)

| 指标 | Doubao-Seed-2.0-Code | MiniMax-M2.5 | Kimi-K2.5 |
|------|---------------------|-------------|-----------|
| 平均 TPS | **86.9** tok/s | 67.0 tok/s | 35.6 tok/s |
| 最小 TPS | **68.1** tok/s | 44.9 tok/s | 32.8 tok/s |
| 最大 TPS | **102.0** tok/s | 91.6 tok/s | 39.6 tok/s |

#### CPS (Characters Per Second)

| 指标 | Doubao-Seed-2.0-Code | MiniMax-M2.5 | Kimi-K2.5 |
|------|---------------------|-------------|-----------|
| 平均 CPS | **260.6** char/s | 201.1 char/s | 106.8 char/s |
| 最小 CPS | **204.4** char/s | 134.8 char/s | 98.4 char/s |
| 最大 CPS | **306.0** char/s | 274.7 char/s | 118.8 char/s |

#### 延迟 & 耗时

| 指标 | Doubao-Seed-2.0-Code | MiniMax-M2.5 | Kimi-K2.5 |
|------|---------------------|-------------|-----------|
| 平均生成时间 | **33.27 s** | 57.65 s | 1.5 min |
| 平均总延迟 | **33.58 s** | 1.6 min | 3.6 min |
| 最大总延迟 | **1.2 min** | 2.6 min | 5.9 min |
| 测试总耗时 | **6.1 min** | 16.1 min | 42.8 min |

#### 输出量

| 指标 | Doubao-Seed-2.0-Code | MiniMax-M2.5 | Kimi-K2.5 |
|------|---------------------|-------------|-----------|
| 总输出 Tokens | 25.6k | 36.6k | 30.3k |
| 总输出字符 | 76.9k | 109.9k | 91.0k |

### 吞吐测试 (并发=2, 4 次请求)

| 指标 | Doubao-Seed-2.0-Code | MiniMax-M2.5 | Kimi-K2.5 |
|------|---------------------|-------------|-----------|
| 成功率 | 4/4 (100%) | 4/4 (100%) | 4/4 (100%) |
| RPS | **1.00** | 0.37 | 0.06 |
| 平均延迟 | **1.92 s** | 4.60 s | 32.39 s |
| P50 延迟 | **1.95 s** | 4.25 s | 30.58 s |
| P95 延迟 | **2.01 s** | 5.97 s | 38.95 s |
| P99 延迟 | **2.01 s** | 6.14 s | 39.98 s |
| 平均 TPS | 68.5 tok/s | **112.5** tok/s | 22.8 tok/s |

### 逐题质量对比

| 任务 | Doubao-Seed-2.0-Code | MiniMax-M2.5 | Kimi-K2.5 |
|------|---------------------|-------------|-----------|
| M01 并发安全的连接池 | 79.0 | **88.0** | 77.0 |
| M03 修复 Go Race Condition | 76.7 | 78.7 | **87.6** |
| M04 代码审查: Node.js 安全 | **72.8** | 67.4 | 66.8 |
| M05 编写 pytest 测试 | **85.9** | 81.8 | 79.2 |
| H01 实现跳表 (Skip List) | 78.4 | **88.9** | 79.1 |
| H03 重构: 消除 God Object | 63.9 | 74.5 | **77.8** |
| X01 Go 并发 Bug 修复 | **78.6** | 73.0 | 72.2 |
| X02 Promise.allSettled + 重试 | 76.7 | 84.7 | **90.9** |
| S02 长代码生成: REST 框架 | **91.0** | 83.0 | 89.0 |

### 逐题 TTFT 对比

| 任务 | Doubao-Seed-2.0-Code | MiniMax-M2.5 | Kimi-K2.5 |
|------|---------------------|-------------|-----------|
| M01 并发安全的连接池 | **345 ms** | 28.49 s | 4.6 min |
| M03 修复 Go Race Condition | **282 ms** | 19.43 s | 1.8 min |
| M04 代码审查: Node.js API | **323 ms** | 11.99 s | 19.51 s |
| M05 编写 pytest 测试 | **294 ms** | 31.92 s | 2.2 min |
| H01 实现跳表 (Skip List) | **256 ms** | 1.0 min | 1.5 min |
| H03 重构: 消除 God Object | **366 ms** | 13.99 s | 38.45 s |
| X01 Go 并发 Bug 修复 | **465 ms** | 1.1 min | 4.3 min |
| X02 Promise.allSettled + 重试 | **284 ms** | 18.74 s | 2.1 min |
| S02 长代码生成: REST 框架 | **234 ms** | 1.2 min | 1.4 min |

### 一致性测试 (量化/掺水检测, 2 轮)

| 任务 | Doubao-Seed-2.0-Code | MiniMax-M2.5 | Kimi-K2.5 |
|------|---------------------|-------------|-----------|
| C01 平均分 | 42.5 | 42.9 | 32.2 |
| C01 标准差 | **8.13** | 0.00 | 0.00 |
| C01 相似度 | 0.29 | 1.00 | 0.37 |
| C01 判定 | **一般** | 稳定 | 稳定 |

> Doubao 一致性标准差 8.13, 判定"一般" — 两次运行结果有差异, 值得关注。

### 分析

1. **Doubao-Seed-2.0-Code 综合第一 (A 级, 80.2 分)**: 速度碾压 (TTFT 317ms, 平均 TPS 86.9), 质量也不差 (78.1)。但**一致性有波动** (标准差 8.13, 评级降为"一般"), 两次跑同一题的结果不完全一致。
2. **Kimi-K2.5 质量并列第一但速度垫底**: 质量分 80.0 与 MiniMax 并列, 在 expert 级任务上表现最好 (X02 得分 90.9, M03 得分 87.6)。但 TTFT 平均 2.1 分钟, 总测试耗时 42.8 分钟 — 原因是内部 reasoning/thinking 阶段消耗大量时间。
3. **MiniMax-M2.5 均衡型选手**: 质量 80.0 并列第一, 一致性完美 (100.0), 吞吐峰值 TPS 112.5 tok/s 是三者最高。但 TTFT 中等 (35.7s), 综合得分被速度拖累。
4. **难题质量差距缩小**: 在 medium/hard/expert 任务上三者质量接近 (78.1~80.0), 远好于 easy 测试时的差距。说明这三个模型在复杂编码任务上的核心能力相当。
5. **Kimi reasoning 模式的代价**: TTFT 中 thinking 阶段预估占 90%+, 如果关闭 reasoning 模式, Kimi 的速度和延迟会有质的提升, 但质量可能下降。

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
