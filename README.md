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

# 只跑新增模型, 结果合并到已有 JSON (按 model_name 去重覆盖)
python -m llm_bench run -c config.yaml -m Kimi-Native-Think-On -m Kimi-Native-Think-Off --merge

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

coding_plans:
  ali:                           # Coding Plan 名称
    name: "阿里云百炼"
    api_type: anthropic          # API 协议 (anthropic / openai)
    base_url: https://coding.dashscope.aliyuncs.com/apps/anthropic
    api_key: "<YOUR_API_KEY>"
    max_tokens: 32768
    timeout: 180
    models:
      - name: Qwen3-Coder-Next    # 可覆盖 Plan 级别任何字段
        model: qwen3-coder-next
        api_type: openai
        base_url: https://coding.dashscope.aliyuncs.com/v1
      - name: Ali-Kimi-Think-On
        model: kimi-k2.5
        thinking: enabled          # thinking 控制
        timeout: 300
```

支持的 API 协议: `openai` (OpenAI 兼容), `anthropic` (Anthropic 兼容) — 通过 `api_type` 字段配置, 可在 Plan 和 Model 级别分别指定

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

## 评测报告: 2026-02-15 — Thinking On/Off 对比 + 原生 Kimi 对比

**测试环境**: macOS, Python 3.9.6, 网络环境 (中国大陆)  
**测试配置**: 4 道 medium/hard/expert 题, 吞吐并发 2, 一致性 1 轮, anti-cache 开启  
**测试时间**: 2026-02-15 21:07 ~ 21:47 (UTC+8), 原生 Kimi 补测 2026-02-16 00:01 ~ 00:10  
**本次重点**: 对比 Thinking 模式开启/关闭对**质量**和**速度**的影响; 方舟 vs 原生 Kimi API 性能差异

### 被测模型 (7 种配置)

| 模型 | 底层 | Thinking | max_tokens | API 来源 | 备注 |
|------|------|----------|------------|----------|------|
| MiniMax-M2.5 | MiniMax-M2.5 | 服务端默认 (On) | 16384 | MiniMax | 基线 |
| Doubao-Default | doubao-seed-2.0-code | Off (不支持) | 32768 | 方舟 | 速度型 |
| Doubao-Think-On | doubao-seed-2.0-code | **enabled** | 32768 | 方舟 | 显式开启 |
| Kimi-Think-On | kimi-k2.5 | **enabled** | 32768 | 方舟 | 方舟版 |
| Kimi-Think-Off | kimi-k2.5 | **disabled** | 32768 | 方舟 | 方舟版 |
| Kimi-Native-Think-On | kimi-k2.5 | **enabled** | 32768 | **原生 Kimi** | api.kimi.com |
| Kimi-Native-Think-Off | kimi-k2.5 | **disabled** | 32768 | **原生 Kimi** | api.kimi.com |

### 测试任务 (4 道)

| ID | 任务 | 难度 | 类别 |
|----|------|------|------|
| M01 | 并发安全的连接池 (asyncio) | Medium | 代码生成 |
| M03 | 修复 Go Race Condition | Medium | Bug 修复 |
| H01 | 实现跳表 (Skip List) | Hard | 算法 |
| X02 | Promise.allSettled + 重试 | Expert | 代码生成 |

### 综合排行

| # | 模型 | 综合 | 质量 | 速度 | 吞吐 | 一致性 | 等级 | 总耗时 |
|---|------|------|------|------|------|--------|------|--------|
| 1 | **Kimi-Native-Think-Off** | **86.7** | **87.6** | **94.2** | 57.9 | 100.0 | **S** | **1.7 min** |
| 2 | Doubao-Default | 81.4 | 81.2 | 82.7 | **61.3** | 100.0 | A | 2.0 min |
| 3 | Doubao-Think-On | 77.8 | 89.0 | 60.0 | 51.9 | 100.0 | B | 12.8 min |
| 4 | Kimi-Native-Think-On | 75.0 | 82.8 | 60.0 | 51.6 | 100.0 | B | 6.8 min |
| 5 | MiniMax-M2.5 | 74.7 | 82.1 | 60.0 | 51.8 | 100.0 | B | 4.7 min |
| 6 | Kimi-Think-Off (方舟) | 72.9 | 82.4 | 50.7 | 54.2 | 100.0 | B | 5.4 min |
| 7 | Kimi-Think-On (方舟) | 69.2 | 81.5 | 40.0 | 50.4 | 100.0 | C | 15.2 min |

### Thinking 模式对比: 豆包

| 指标 | Doubao-Default (Off) | Doubao-Think-On | 变化 |
|------|---------------------|-----------------|------|
| 质量 | 81.2 | **89.0** | **+7.8 (+9.6%)** |
| 平均 TTFT | **327 ms** | 2.2 min | 慢 400x |
| 平均 TPS | 72.5 | **406.9** | 快 5.6x (含 thinking tokens) |
| 平均 CPS | **299.9** | 294.7 | 持平 |
| 总耗时 | **2.0 min** | 12.8 min | 慢 6.4x |
| Thinking Tokens | 0 | 47.5k | - |

> 豆包开 thinking 后质量**显著提升** (+9.6%), 代价是耗时增加 6 倍。Thinking 对**难题帮助最大**: H01 跳表 90.7→95.4, M03 Race Condition 78.7→88.2。

### Thinking 模式对比: Kimi

| 指标 | Kimi-Think-Off | Kimi-Think-On | 变化 |
|------|---------------|---------------|------|
| 质量 | **82.4** | 81.5 | **-0.9 (持平!)** |
| 平均 TTFT | **178 ms** | 1.8 min | 慢 600x |
| 平均 TPS | 22.0 | 68.4 | 3x (含 thinking tokens) |
| 平均 CPS | 99.3 | 110.9 | +12% |
| 总耗时 | **5.4 min** | 15.2 min | 慢 2.8x |
| Thinking Tokens | 0 | 14.8k | - |

> Kimi (方舟) 关 thinking 后质量**基本不变** (82.4 vs 81.5), 速度快了 **3 倍**。Thinking 在这些任务上投入产出比很低。

### 原生 Kimi API vs 方舟 Kimi: 同一个模型, 天壤之别

| 指标 | 方舟 Think-Off | **原生 Think-Off** | 方舟 Think-On | **原生 Think-On** |
|------|---------------|-------------------|--------------|------------------|
| 综合 | 72.9 | **86.7** (+18.9%) | 69.2 | **75.0** (+8.4%) |
| 质量 | 82.4 | **87.6** (+6.3%) | 81.5 | **82.8** (+1.6%) |
| 速度 | 50.7 | **94.2** (+86%) | 40.0 | **60.0** (+50%) |
| TTFT | 178 ms | **711 ms** | 1.8 min | **46.5 s** |
| CPS (字符/秒) | 99.3 | **520.2** (5.2x!) | 110.9 | **507.6** (4.6x!) |
| 总耗时 | 5.4 min | **1.7 min** (3.2x 快) | 15.2 min | **6.8 min** (2.2x 快) |
| Thinking 耗时 | — | — | 2.7 min | **58.7 s** (2.8x 快) |

> **同一个 kimi-k2.5 模型, 原生 API (api.kimi.com) 的 CPS 是方舟 API 的 5 倍!** 质量也更高。原生 Kimi-Think-Off 以 86.7 分**登顶全部 7 个模型的榜首**, 超越此前的冠军 Doubao-Default (81.4)。方舟版 Kimi 很可能存在限速或量化, 严重影响了性能表现。

### 原生 Kimi Thinking On/Off 对比

| 指标 | 原生 Think-Off | 原生 Think-On | 变化 |
|------|---------------|---------------|------|
| 质量 | **87.6** | 82.8 | **-4.8 (关比开好!)** |
| 平均 TTFT | **711 ms** | 46.5 s | 慢 65x |
| 平均 CPS | 520.2 | **507.6** | 持平 |
| 总耗时 | **1.7 min** | 6.8 min | 慢 4x |
| Thinking Tokens | 0 | 29.0k | — |

> 原生 Kimi 同样证实: **关 thinking 质量更高** (87.6 vs 82.8)。Think-Off 在 4 道题中 3 道胜出, 且速度快 4 倍。

### 客观指标对比

#### TTFT (首字延迟)

| 指标 | Doubao-Default | Doubao-Think | MiniMax | Kimi-Off (方舟) | Kimi-On (方舟) | **Native-Off** | **Native-On** |
|------|---------------|-------------|---------|----------|---------|------------|-----------|
| 平均 | **327 ms** | 2.2 min | 28.29 s | 178 ms | 1.8 min | 711 ms | 46.52 s |
| 最小 | **272 ms** | 1.1 min | 13.05 s | 136 ms | 1.3 min | 546 ms | 32.20 s |
| 最大 | 418 ms | 2.9 min | 53.29 s | **224 ms** | 2.1 min | 927 ms | 65.58 s |

#### TPS & CPS

| 指标 | Doubao-Default | Doubao-Think | MiniMax | Kimi-Off (方舟) | Kimi-On (方舟) | **Native-Off** | **Native-On** |
|------|---------------|-------------|---------|----------|---------|------------|-----------|
| 平均 TPS | 72.5 | 406.9 | 116.0 | 22.0 | 68.4 | 114.0 | **489.0** |
| 平均 CPS | 299.9 | 294.7 | 290.2 | 99.3 | 110.9 | **520.2** | 507.6 |

> TPS 包含 thinking tokens, 因此 Think-On 的 TPS 虚高。**CPS (字符/秒) 更能反映真实输出速度**。原生 Kimi CPS **520** 远超方舟版 Kimi (99-111) 和 Doubao/MiniMax (290)。

#### 延迟 & 耗时

| 指标 | Doubao-Default | Doubao-Think | MiniMax | Kimi-Off (方舟) | Kimi-On (方舟) | **Native-Off** | **Native-On** |
|------|---------------|-------------|---------|----------|---------|------------|-----------|
| 平均生成时间 | 19.71 s | 26.86 s | 24.88 s | 1.2 min | 56.51 s | **18.02 s** | 12.92 s |
| 平均总延迟 | 20.04 s | 2.7 min | 53.17 s | 1.2 min | 2.7 min | **18.73 s** | 59.44 s |
| 测试总耗时 | 2.0 min | 12.8 min | 4.7 min | 5.4 min | 15.2 min | **1.7 min** | 6.8 min |

#### Thinking 阶段统计

| 指标 | Doubao-Think | MiniMax | Kimi-On (方舟) | **Native-On** |
|------|-------------|---------|---------|-----------|
| 平均 Think 耗时 | 2.7 min | 47.51 s | 2.7 min | **58.74 s** |
| 最大 Think 耗时 | 3.4 min | 1.3 min | 3.3 min | **1.4 min** |
| Think 总 Tokens | 47.5k | 6,680 | 14.8k | **29.0k** |
| 涉及 Thinking 题数 | 4/4 | 4/4 | 4/4 | 4/4 |

### 逐题质量对比

| 任务 | Doubao-Default | Doubao-Think | MiniMax | Kimi-Off (方舟) | Kimi-On (方舟) | **Native-Off** | **Native-On** |
|------|---------------|-------------|---------|----------|---------|------------|-----------|
| M01 并发安全的连接池 | 79.0 | 86.0 | 85.0 | 84.0 | 72.0 | **87.0** | 82.0 |
| M03 修复 Race Condition | 78.7 | 88.2 | 69.2 | 78.7 | 82.3 | 82.3 | **91.0** |
| H01 实现跳表 (Skip List) | 90.7 | **95.4** | 89.9 | 81.4 | 87.3 | 90.7 | 77.3 |
| X02 Promise.allSettled | 76.3 | 86.3 | 84.4 | 85.6 | 84.3 | **90.6** | 81.0 |

### 逐题 TTFT 对比

| 任务 | Doubao-Default | Doubao-Think | MiniMax | Kimi-Off (方舟) | Kimi-On (方舟) | **Native-Off** | **Native-On** |
|------|---------------|-------------|---------|----------|---------|------------|-----------|
| M01 并发安全的连接池 | **328 ms** | 2.8 min | 27.72 s | 224 ms | 2.1 min | 927 ms | 65.58 s |
| M03 修复 Race Condition | **418 ms** | 1.1 min | 13.05 s | 136 ms | 1.3 min | 694 ms | 37.04 s |
| H01 实现跳表 (Skip List) | **292 ms** | 2.1 min | 53.29 s | 147 ms | 1.7 min | 546 ms | 32.20 s |
| X02 Promise.allSettled | **272 ms** | 2.9 min | 19.09 s | 204 ms | 1.9 min | 680 ms | 51.24 s |

### 吞吐测试 (并发=2)

| 指标 | Doubao-Default | Doubao-Think | MiniMax | Kimi-Off (方舟) | Kimi-On (方舟) | **Native-Off** | **Native-On** |
|------|---------------|-------------|---------|----------|---------|------------|-----------|
| RPS | **1.13** | 0.19 | 0.18 | 0.42 | 0.04 | 0.79 | 0.16 |
| 平均延迟 | **1.60 s** | 7.20 s | 8.70 s | 4.71 s | 34.19 s | 2.44 s | 12.30 s |
| P95 延迟 | **1.76 s** | 9.98 s | 10.82 s | 4.73 s | 46.97 s | 2.53 s | 12.34 s |
| 平均 TPS | 84.7 | 327.8 | **362.2** | 21.6 | 180.0 | 56.7 | 41.6 |

### 分析与结论

1. **原生 Kimi-Think-Off 登顶**: 综合 86.7 (S 级), 质量 87.6, 速度 94.2, 总耗时仅 1.7 分钟。同一个 kimi-k2.5 模型, 原生 API 的 CPS 是方舟版的 **5 倍** (520 vs 99), 质量也高 5 分。**原生 Kimi 是目前最强编码模型配置**。

2. **方舟 Kimi 疑似限速/量化**: 同一个 kimi-k2.5 模型, 方舟版 CPS 仅 99-111 字符/秒, 原生版 507-520 字符/秒, 差距 5 倍。质量也有明显差距 (方舟 82.4 vs 原生 87.6)。如果你在用方舟的 Kimi, **强烈建议切换到原生 API**。

3. **Kimi Thinking 不值得开**: 不管方舟还是原生, Kimi K2.5 开 thinking 后质量**不升反降** (原生: 87.6→82.8, 方舟: 82.4→81.5)。代价是 3-4 倍耗时。**Kimi K2.5 务必关闭 thinking**。

4. **豆包 Thinking 是"真提升"**: 开启后质量从 81.2 跳到 **89.0** (+9.6%), 4 道题全面提升。代价是 TTFT 从 327ms 涨到 2.2 min, 总耗时 6 倍。**适合追求最高代码质量的场景**。

5. **MiniMax 受 thinking 拖累**: 服务端默认开启 thinking (无法关闭), Think 平均耗时 47.51s, 导致 TTFT 28.29s。质量 82.1, 如能关 thinking 可能更好。

### max_tokens 上限实测

| 模型 | 8192 | 16384 | 32768 | 65536 | 实际上限 |
|------|------|-------|-------|-------|---------|
| Doubao-Seed-2.0-Code | ✅ | ✅ | ✅ | ✅ | **≥ 64K** |
| Kimi-K2.5 | ✅ | ✅ | ✅ | ❌ 400 | **32K** |
| MiniMax-M2.5 | ✅ | ✅ | ✅ | ✅ | **≥ 64K** |

### Thinking 模式可控性

| 模型 | 默认 Thinking | 可否关闭 | 控制方式 | 来源 |
|------|-------------|---------|---------|------|
| Doubao-Seed-2.0-Code | **Off** | 可开启 | `thinking: enabled` | API 实测 |
| Kimi-K2.5 | **On** | **可关闭** | `thinking: disabled` | [Kimi 官方文档](https://platform.moonshot.cn/docs/api/chat) |
| MiniMax-M2.5 | **On** | 未知 | 服务端默认, 暂无参数关闭 | API 实测 |

---

## 评测报告: 2026-02-28 — 新增 Qwen3.5-Plus / CUCloud GLM-5

**测试环境**: macOS, Python 3.14.3, 网络环境 (中国大陆)  
**测试配置**: 4 道 medium/hard/expert 题, 吞吐并发 2, 一致性 1 轮, anti-cache 开启  
**测试时间**: 2026-02-28 17:30 ~ 18:40 (UTC+8)  
**本次重点**: 新增阿里云百炼 Qwen3.5-Plus 和联通云 CUCloud GLM-5, 对比全系 9 个模型

### 新增模型

| 模型 | 底层 | Provider | API 兼容 | max_tokens | 备注 |
|------|------|----------|----------|------------|------|
| Qwen3.5-Plus | qwen3.5-plus | 阿里云百炼 DashScope | OpenAI | 32768 | 有服务端 reasoning (TTFT 长) |
| CUCloud-GLM-5 | glm-5 | 联通云 CUCloud | OpenAI | 16384 | Z.ai GLM-5, 含 inline thinking |

> **注**: Qwen3-Coder-Plus 在首轮评测中得分 82.6 (A), 但因进程中断数据未持久化, 后因 DashScope 账户欠费无法复测。

### 全系综合排行 (9 个模型)

| # | 模型 | 综合 | 质量 | 速度 | 吞吐 | 一致性 | 等级 | 总耗时 |
|---|------|------|------|------|------|--------|------|--------|
| 1 | **Kimi-Native-Think-Off** | **86.7** | 87.6 | **94.2** | **57.9** | 100.0 | **S** | **1.7 min** |
| 2 | Doubao-Default | 81.4 | 81.2 | 82.7 | 61.3 | 100.0 | A | 2.0 min |
| 3 | Doubao-Think-On | 77.8 | 89.0 | 60.0 | 51.9 | 100.0 | B | 12.8 min |
| 4 | Kimi-Native-Think-On | 75.0 | 82.8 | 60.0 | 51.6 | 100.0 | B | 6.8 min |
| 5 | MiniMax-M2.5 | 74.7 | 82.1 | 60.0 | 51.8 | 100.0 | B | 4.7 min |
| 6 | **Qwen3.5-Plus** | **73.5** | 82.1 | 56.2 | 50.3 | 100.0 | B | 8.3 min |
| 7 | Kimi-Think-Off (方舟) | 72.9 | 82.4 | 50.7 | 54.2 | 100.0 | B | 5.4 min |
| 8 | **CUCloud-GLM-5** | **72.0** | **93.1** | 30.2 | 50.0 | 100.0 | B | 29.2 min |
| 9 | Kimi-Think-On (方舟) | 69.2 | 81.5 | 40.0 | 50.4 | 100.0 | C | 15.2 min |

### Qwen3.5-Plus 详情

| 指标 | 值 |
|------|-----|
| 平均 TTFT | 59.9 s (服务端 reasoning 导致, 8.1s ~ 87.0s) |
| 平均 TPS | 93.9 |
| 平均 CPS | 281.9 |
| 平均生成时间 | 23.7 s |
| 平均总延迟 | 83.6 s |
| 总输出 Tokens | 8,958 |
| 吞吐 RPS (并发=2) | 0.03 |

逐题质量: M01=79.0, M03=83.7, H01=85.3, X02=80.6

> Qwen3.5-Plus 的 **生成速度不错** (CPS 281.9, 接近 Doubao), 但 **TTFT 非常高** (平均 60s), 说明有服务端 reasoning/thinking 过程。质量 82.1 与 Doubao-Default / Kimi-Think-Off 相当。

### CUCloud GLM-5 详情

| 指标 | 值 |
|------|-----|
| 平均 TTFT | 3.2 s (1.7s ~ 6.4s) |
| 平均 TPS | 36.6 |
| 平均 CPS | 109.9 |
| 平均生成时间 | 4.7 min |
| 平均总延迟 | 4.8 min |
| 总输出 Tokens | 39,730 (含 inline thinking) |
| 吞吐 RPS (并发=2) | ~0 |

逐题质量: M01=**91.0**, M03=**88.2**, H01=**97.7**, X02=**95.6**

> CUCloud GLM-5 质量 **93.1 (S 级)**, 是全部 9 个模型中**质量最高**的。H01 跳表得分 97.7 远超所有其他模型。但速度极慢 (TPS 仅 36.6, 平均每题 4.8 分钟), 且模型将 thinking 过程 inline 输出到 content 中 (导致 token 数虚高 39.7k)。适合**对质量要求极高、不赶时间**的场景。

### 逐题质量对比 (新增 vs 已有)

| 任务 | Kimi-Native-Off | Doubao-Think | **Qwen3.5-Plus** | **CUCloud-GLM-5** |
|------|----------------|-------------|-----------------|------------------|
| M01 并发安全的连接池 | 87.0 | 86.0 | 79.0 | **91.0** |
| M03 修复 Race Condition | 82.3 | 88.2 | 83.7 | **88.2** |
| H01 实现跳表 (Skip List) | 90.7 | 95.4 | 85.3 | **97.7** |
| X02 Promise.allSettled | 90.6 | 86.3 | 80.6 | **95.6** |

### 技术发现

1. **Python 3.14 + anyio TLS 兼容性问题**: anyio 4.12.1 的 `TLSStream.wrap` 对部分 HTTPS 服务器 (dashscope.aliyuncs.com 等) 握手失败, 原因是 STARTTLS 方式在某些服务器上触发 `EndOfStream`。解决方案: 自定义 httpcore 网络后端 (`_async_backend.py`), 使用 asyncio 原生 SSL 连接。

2. **CUCloud 使用 OpenAI 兼容 API**: 联通云 CUCloud 的 `/v1/chat/completions` 端点使用 `Authorization: Bearer` 认证, 是 OpenAI 兼容格式 (非 Anthropic)。

3. **CUCloud GLM-5 inline thinking**: 模型的推理过程直接输出在 `content` 字段中 (以 `</think>` 分隔), 不像 Anthropic 的 `thinking` block 分离。导致输出 token 数虚高。

---

## 评测报告: 2026-03-06 — 阿里云百炼 Coding Plan 全系模型 + 新增 Qwen3-Coder-Next

**测试环境**: macOS, Python 3.14.3, 网络环境 (中国大陆)  
**测试配置**: 4 道 medium/hard/expert 题, 吞吐并发 2, 一致性 1 轮, anti-cache 开启  
**测试时间**: 2026-03-06 10:20 ~ 11:50 (UTC+8)  
**本次重点**: 阿里云百炼 Coding Plan 全系 9 个模型评测 (含新增 Qwen3-Coder-Next / Qwen3-Max / GLM-4.7); 修复 Anthropic SDK auth_token 环境变量冲突

### 新增模型

| 模型 | 底层 | API 兼容 | max_tokens | 备注 |
|------|------|----------|------------|------|
| Qwen3-Coder-Next | qwen3-coder-next | OpenAI | 32768 | 新模型, 速度极快 (CPS 947) |
| Qwen3-Max | qwen3-max-2026-01-23 | OpenAI | 32768 | 深度思考模型 |
| Ali-GLM-5 | glm-5 | Anthropic | 32768 | 智谱 GLM-5, 含深度思考 |
| Ali-GLM-4.7 | glm-4.7 | Anthropic | 32768 | 智谱 GLM-4.7, 含深度思考 |
| Ali-Kimi-Think-On | kimi-k2.5 | Anthropic | 32768 | Kimi K2.5, thinking 显式开启 |
| Ali-Kimi-Think-Off | kimi-k2.5 | Anthropic | 32768 | Kimi K2.5, thinking 显式关闭 |
| Ali-MiniMax-M2.5 | MiniMax-M2.5 | Anthropic | 32768 | MiniMax M2.5, thinking 默认开启 |

> **注**: Qwen3-Coder-Plus 和 Qwen3.5-Plus 也通过 Coding Plan 重新评测, 更新了结果。

### 全系综合排行 (17 个模型)

| # | 模型 | 综合 | 质量 | 速度 | 吞吐 | 一致性 | 等级 | 总耗时 |
|---|------|------|------|------|------|--------|------|--------|
| 1 | **Qwen3-Coder-Next** | **89.7** | 85.9 | **100.0** | **73.7** | 100.0 | **A** | **43s** |
| 2 | Kimi-Native-Think-Off | 86.7 | 87.6 | 94.2 | 57.9 | 100.0 | S | 1.7 min |
| 3 | Qwen3-Coder-Plus | 81.9 | 76.0 | 95.1 | 59.8 | 100.0 | A | 1.6 min |
| 4 | Doubao-Default | 81.4 | 81.2 | 82.7 | 61.3 | 100.0 | A | 2.0 min |
| 5 | Doubao-Think-On | 77.8 | 89.0 | 60.0 | 51.9 | 100.0 | B | 12.8 min |
| 6 | **Ali-MiniMax-M2.5** | **75.8** | **86.8** | 54.2 | 54.5 | 100.0 | B | 3.9 min |
| 7 | Kimi-Native-Think-On | 75.0 | 82.8 | 60.0 | 51.6 | 100.0 | B | 6.8 min |
| 8 | **Ali-Kimi-Think-On** | **75.0** | 83.0 | 60.0 | 51.0 | 100.0 | B | 11.4 min |
| 9 | MiniMax-M2.5 | 74.7 | 82.1 | 60.0 | 51.8 | 100.0 | B | 4.7 min |
| 10 | **Ali-Kimi-Think-Off** | **74.0** | **86.9** | 46.5 | 55.4 | 100.0 | B | 4.1 min |
| 11 | Kimi-Think-Off (方舟) | 72.9 | 82.4 | 50.7 | 54.2 | 100.0 | B | 5.4 min |
| 12 | **Qwen3.5-Plus** | **72.2** | 83.2 | 48.9 | 50.3 | 100.0 | B | 10.2 min |
| 13 | **Ali-GLM-5** | **72.2** | 76.8 | 60.0 | 51.0 | 100.0 | B | 16.4 min |
| 14 | CUCloud-GLM-5 | 72.0 | 93.1 | 30.2 | 50.0 | 100.0 | B | 29.2 min |
| 15 | **Qwen3-Max** | **71.5** | 75.1 | 58.2 | 54.6 | 100.0 | B | 2.5 min |
| 16 | **Ali-GLM-4.7** | **71.4** | 74.6 | 60.0 | 52.3 | 100.0 | B | 10.6 min |
| 17 | Kimi-Think-On (方舟) | 69.2 | 81.5 | 40.0 | 50.4 | 100.0 | C | 15.2 min |

### 阿里云百炼 Coding Plan 模型对比

| # | 模型 | 综合 | 质量 | 速度 | CPS | TTFT | 总耗时 |
|---|------|------|------|------|-----|------|--------|
| 1 | **Qwen3-Coder-Next** | **89.7** | 85.9 | **100.0** | **946.8** | 465ms | **43s** |
| 2 | Qwen3-Coder-Plus | 81.9 | 76.0 | 95.1 | 306.1 | 633ms | 1.6 min |
| 3 | Ali-MiniMax-M2.5 | 75.8 | **86.8** | 54.2 | 245.4 | 16.1s | 3.9 min |
| 4 | Ali-Kimi-Think-On | 75.0 | 83.0 | 60.0 | 237.8 | 1.6 min | 11.4 min |
| 5 | Ali-Kimi-Think-Off | 74.0 | 86.9 | 46.5 | 174.9 | 1.2s | 4.1 min |
| 6 | Qwen3.5-Plus | 72.2 | 83.2 | 48.9 | 247.3 | 1.4 min | 10.2 min |
| 7 | Ali-GLM-5 | 72.2 | 76.8 | 60.0 | 205.1 | 2.6 min | 16.4 min |
| 8 | Qwen3-Max | 71.5 | 75.1 | 58.2 | 159.5 | 1.1s | 2.5 min |
| 9 | Ali-GLM-4.7 | 71.4 | 74.6 | 60.0 | 285.3 | 1.6 min | 10.6 min |

### 逐题质量对比 (百炼 Coding Plan)

| 任务 | Qwen3-Coder-Next | Qwen3-Coder-Plus | Qwen3.5-Plus | Qwen3-Max | Ali-GLM-5 | Ali-GLM-4.7 | Ali-Kimi-On | Ali-Kimi-Off | Ali-MiniMax |
|------|-------------------|-------------------|--------------|-----------|-----------|-------------|-------------|--------------|-------------|
| M01 并发安全的连接池 | 81.0 | 64.0 | **89.0** | 64.0 | 70.0 | 65.2 | 81.0 | 79.0 | 83.0 |
| M03 修复 Race Condition | 83.7 | 80.9 | 83.7 | 77.5 | 79.9 | 67.8 | **86.0** | 84.9 | 79.8 |
| H01 实现跳表 (Skip List) | 89.6 | 77.3 | 81.4 | 77.3 | 76.1 | 77.3 | 77.3 | **90.7** | **90.7** |
| X02 Promise.allSettled | 89.3 | 81.7 | 78.7 | 81.7 | 81.3 | 88.0 | 87.9 | 92.9 | **93.7** |

### 分析与结论

1. **Qwen3-Coder-Next 登顶全榜**: 综合 89.7 (A 级), 首次超越 Kimi-Native-Think-Off (86.7)。CPS 高达 **946.8 字符/秒**, 是所有模型中最快的, 比原生 Kimi (520) 快 82%。质量 85.9 也很出色。总耗时仅 43 秒, 是最快完成全部测试的模型。

2. **百炼 Coding Plan 性价比极高**: 一个套餐同时涵盖千问、Kimi、GLM、MiniMax 四大品牌模型。Ali-Kimi-Think-Off (86.9) 和 Ali-MiniMax-M2.5 (86.8) 的质量分甚至超过了各自原生 API 的表现。

3. **百炼 vs 原生 Kimi 对比**: Ali-Kimi-Think-Off 质量 86.9 vs 原生 Kimi-Think-Off 87.6, 差距缩小到 0.7 分。但 CPS 差距明显 (174.9 vs 520.2), 百炼的 Kimi 仍有限速。

4. **GLM 系列表现中规中矩**: GLM-5 和 GLM-4.7 综合均在 71-72, 质量 74-77, 低于同为百炼的 Kimi 和 MiniMax。深度思考耗时较长 (TTFT 1.6~2.6 min)。

5. **Qwen3-Max 意外垫底**: 综合 71.5, 在千问系列中最低。TPS 仅 53.2, 远低于 Qwen3-Coder-Next (315.6) 和 Qwen3-Coder-Plus (98.2)。定位为通用模型, 编码能力不如专用 coder 系列。

### 技术发现: Anthropic SDK auth_token 环境变量冲突

Anthropic Python SDK v0.84.0 在初始化时会自动读取 `ANTHROPIC_AUTH_TOKEN` 环境变量, 并在请求头中同时发送 `X-Api-Key` 和 `Authorization: Bearer <auth_token>`。当在 Amp/Claude Code 环境中运行时, 该环境变量会被自动设置, 导致第三方 Anthropic 兼容端点 (如阿里云百炼) 优先读取无效的 `Authorization` 头而返回 401 错误。

修复方案: 在 `AnthropicCompatProvider` 中, 创建客户端后显式设置 `self.client.auth_token = None`, 清除从环境变量继承的无效 token。

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
│   │   ├── registry.py      # Provider 预设注册表
│   │   └── _async_backend.py  # asyncio 网络后端 (修复 Python 3.14 TLS)
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
