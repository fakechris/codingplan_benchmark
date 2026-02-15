# 实测 Kimi-K2.5 / 豆包 Code / MiniMax-M2.5：Thinking 模式该不该开？原生 Kimi 比方舟快 5 倍！

> 写了个 benchmark 工具，把三家的编码模型扒了个底朝天。还发现同一个 Kimi 模型，原生 API 和方舟 API 差距惊人。

---

## 背景

最近在给团队选 coding copilot 的底座模型。方舟平台上的 Kimi-K2.5 和 doubao-seed-2.0-code 都走 Anthropic 兼容协议，MiniMax 的 M2.5 也是。三家 API 格式几乎一样，正好拉到一起比。

但跑了第一轮就发现问题：Kimi 的 TTFT（首字延迟）动辄两三分钟，MiniMax 也要半分钟。一查，原来两家默认开了 thinking/reasoning 模式——模型在吐第一个字之前，先在那默默"想"一大段。

这就引出了两个实际问题：
1. **thinking 模式对编码质量到底有多大帮助？值不值得用 3~6 倍的时间去换？**
2. **同一个模型，走方舟中转和走原生 API，性能到底差多少？**

于是写了个工具，做了两组控制变量实验。

---

## 测试工具

开源项目：[codingplan_benchmark](https://github.com/fakechris/codingplan_benchmark)

一句话描述：面向 coding plan 场景的 LLM benchmark，测的不是刷题，是你让模型"帮你写代码"时的真实体验。

### 设计思路

市面上的 benchmark 要么是选择题（MMLU 之类），要么是 HumanEval 那种短函数补全。但实际干活时，你扔给模型的是"帮我实现一个带重试的 Promise.allSettled"、"这段 Go 代码有 race condition，帮我修"这种级别的需求。

所以我做了 19 道任务，覆盖：

- **代码生成**：asyncio 连接池、REST 框架、Promise 封装
- **Bug 修复**：Go race condition、并发死锁
- **算法**：跳表 (Skip List) 实现
- **重构**：消除 God Object
- **代码审查**：Node.js API 安全审计
- **测试编写**：pytest 用例生成

语言覆盖 Python / Go / TypeScript，难度从 Medium 到 Expert。

### 评测维度

| 维度 | 权重 | 怎么测 |
|------|------|--------|
| 质量 | 45% | 规则打分（结构、正确性、边界处理）+ 可选 LLM Judge |
| 速度 | 25% | TTFT + TPS 综合 |
| 吞吐 | 15% | 并发 RPS、P95 延迟 |
| 一致性 | 15% | 多次跑同一题，检测是否"掺水"或量化 |

### 关键指标

不是只看"快不快"，而是拆得很细：

- **TTFT**（首字延迟）：发请求到拿到第一个 token
- **TPS**（Tokens/s）：去掉 TTFT 后的纯生成速度
- **CPS**（Characters/s）：每秒输出字符——这个比 TPS 更真实，因为不同 tokenizer 切出来的 token 长度不一样
- **Thinking Time**：reasoning 模型的"思考阶段"单独计时，不混进 TTFT
- **Thinking Tokens**：思考阶段吃掉了多少 token（这些 token 你看不到，但你在等）

### 反作弊

`anti_cache` 模式开启后，每道题会注入随机 nonce，防止服务端对常见 prompt 做缓存命中。

---

## 实验设计

做了两组对比实验，共 **7 种模型配置**：

**第一组：Thinking On/Off 对比（5 种，全走方舟 API）**

| 配置 | 底层模型 | Thinking | max_tokens |
|------|----------|----------|------------|
| Doubao-Default | doubao-seed-2.0-code | Off | 32768 |
| Doubao-Think-On | doubao-seed-2.0-code | **显式开启** | 32768 |
| Kimi-Think-On | kimi-k2.5 | **显式开启** | 32768 |
| Kimi-Think-Off | kimi-k2.5 | **显式关闭** | 32768 |
| MiniMax-M2.5 | MiniMax-M2.5 | 服务端默认 On | 16384 |

**第二组：原生 Kimi API vs 方舟 API（2 种新增）**

| 配置 | 底层模型 | Thinking | API 来源 |
|------|----------|----------|----------|
| Kimi-Native-Think-On | kimi-k2.5 | **显式开启** | api.kimi.com（原生） |
| Kimi-Native-Think-Off | kimi-k2.5 | **显式关闭** | api.kimi.com（原生） |

测试任务选了 4 道有代表性的（Medium / Hard / Expert 各覆盖）：

1. **M01** — Python asyncio 并发安全连接池（中等 / 代码生成）
2. **M03** — 修复 Go Race Condition（中等 / Bug 修复）
3. **H01** — 实现跳表 Skip List（困难 / 算法）
4. **X02** — Promise.allSettled + 重试机制（专家 / 代码生成）

同一套题，同一环境，同一时间段跑完。中国大陆网络，macOS。

---

## 结果：先看排行

| # | 模型配置 | 综合 | 质量 | 速度 | 等级 | 总耗时 |
|---|----------|------|------|------|------|--------|
| 1 | **Kimi-Native-Think-Off** 🏆 | **86.7** | **87.6** | **94.2** | **S** | **1.7 min** |
| 2 | Doubao-Default | 81.4 | 81.2 | 82.7 | A | 2.0 min |
| 3 | Doubao-Think-On | 77.8 | 89.0 | 60.0 | B | 12.8 min |
| 4 | Kimi-Native-Think-On | 75.0 | 82.8 | 60.0 | B | 6.8 min |
| 5 | MiniMax-M2.5 | 74.7 | 82.1 | 60.0 | B | 4.7 min |
| 6 | Kimi-Think-Off（方舟）| 72.9 | 82.4 | 50.7 | B | 5.4 min |
| 7 | Kimi-Think-On（方舟）| 69.2 | 81.5 | 40.0 | C | 15.2 min |

第一眼看完：**原生 Kimi 关 thinking 直接 S 级登顶**，综合 86.7，质量、速度双杀。

之前 Doubao Default 的 A 级第一被拉下来了。

更有意思的是：同样的 kimi-k2.5 模型，方舟版排第 6，原生版排第 1，差了 **13.8 分**。

---

## 核心发现 1：豆包的 Thinking 是"真提升"

| 指标 | Off | On | 变化 |
|------|-----|-----|------|
| 质量 | 81.2 | **89.0** | +9.6% |
| TTFT | **327ms** | 2.2min | 慢 400 倍 |
| 总耗时 | **2.0min** | 12.8min | 慢 6.4 倍 |
| Think Tokens | 0 | 47,500 | — |

开了 thinking，豆包 4 道题**全线涨分**：

- 跳表实现：90.7 → **95.4**（+4.7）
- Race Condition 修复：78.7 → **88.2**（+9.5）
- Promise 封装：76.3 → **86.3**（+10.0）
- 连接池：79.0 → **86.0**（+7.0）

注意，thinking 阶段吃掉了 **47,500 tokens**——这些你在输出里看不到，但每个 token 都在消耗推理算力和你的等待时间。

**结论：豆包的 thinking 模式有真实的质量提升，尤其在难题上。代价是 6 倍耗时。如果你在做 code review 或者复杂架构设计，开它；日常写 CRUD，别开。**

---

## 核心发现 2：Kimi 的 Thinking 是"空转"

| 指标 | Off | On | 变化 |
|------|-----|-----|------|
| 质量 | **82.4** | 81.5 | -0.9 |
| TTFT | **178ms** | 1.8min | 慢 600 倍 |
| 总耗时 | **5.4min** | 15.2min | 慢 2.8 倍 |
| Think Tokens | 0 | 14,800 | — |

Kimi K2.5 开 thinking 后，质量**没涨反跌**（虽然差异在误差范围内）。但时间直接多了 10 分钟。

逐题看：

- 连接池：84.0 → 72.0（**反而跌了 12 分**）
- Race Condition：78.7 → 82.3（微涨）
- 跳表：81.4 → 87.3（涨了，但不及豆包 95.4）
- Promise：85.6 → 84.3（持平）

14,800 个 thinking tokens 花出去了，质量原地踏步。

**结论：Kimi K2.5 做编码任务时，强烈建议关闭 thinking。TTFT 从 1.8 分钟降到 178ms，质量基本不变。这是免费的 600 倍加速。**

Kimi 官方文档确认可以通过 `thinking: { type: "disabled" }` 关闭。

---

## 核心发现 3：原生 Kimi 比方舟快 5 倍，质量也更高

这是最意外的发现。同一个 kimi-k2.5 模型，走不同 API 入口，性能天差地别：

| 指标 | 方舟 Think-Off | **原生 Think-Off** | 差距 |
|------|---------------|-------------------|------|
| 综合 | 72.9 | **86.7** | +13.8 |
| 质量 | 82.4 | **87.6** | +5.2 |
| TTFT | **178ms** | 711ms | 原生略慢（但都在秒级） |
| CPS（字符/秒）| 99.3 | **520.2** | **快 5.2 倍** |
| 总耗时 | 5.4 min | **1.7 min** | 快 3.2 倍 |

Think-On 也是一样：

| 指标 | 方舟 Think-On | **原生 Think-On** | 差距 |
|------|-------------|-----------------|------|
| CPS | 110.9 | **507.6** | **快 4.6 倍** |
| Thinking 耗时 | 2.7 min（平均） | **58.7 s** | 快 2.8 倍 |
| 总耗时 | 15.2 min | **6.8 min** | 快 2.2 倍 |

方舟版的 Kimi 只能跑到 **100 字符/秒**，原生版跑到 **520 字符/秒**。方舟很可能对 Kimi 做了限速或者用了量化版本。

**如果你正在通过方舟调 Kimi K2.5 的 API，立刻切到 api.kimi.com。这是免费的 5 倍加速 + 5 分质量提升。**

原生 API 用法很简单，Kimi 会员页面生成 API Key，base_url 填 `https://api.kimi.com/coding/`，走的还是 Anthropic 兼容协议，代码不用改。

---

## 核心发现 4：MiniMax 被 Thinking 拖累，但你关不掉

MiniMax M2.5 的 thinking 是**服务端默认开启**的，目前 API 没有暴露关闭参数。

| 指标 | MiniMax (默认 On) |
|------|-------------------|
| 平均 Think 耗时 | 47.51s |
| 最大 Think 耗时 | 1.3min |
| Think 总 Tokens | 6,680 |
| 平均 TTFT | 28.29s |
| 质量 | 82.1 |

质量 82.1 和 Kimi-Think-Off 的 82.4 接近，但 TTFT 28 秒 vs 178ms——**如果 MiniMax 能关 thinking，TTFT 可能也能降到百毫秒级**。

这是一个产品侧的遗憾。希望 MiniMax 后续能开放这个控制参数。

---

## 速度深挖：CPS 比 TPS 更真实

很多评测喜欢比 TPS（Tokens Per Second），但有个坑：**thinking tokens 也算 TPS**。

Doubao-Think-On 的 TPS 高达 406.9 tok/s，看着吓人，但里面 47,500 个 thinking tokens 你是看不到的。真正的输出速度看 **CPS（Characters Per Second）**：

| 模型 | TPS | CPS | CPS 才是真实输出速度 |
|------|-----|-----|---------------------|
| **Kimi-Native-Off** | 114.0 | **520.2** | 🏆 最快 |
| **Kimi-Native-On** | 489.0 | **507.6** | TPS 虚高, CPS 仍最快 |
| Doubao-Default | 72.5 | 299.9 | — |
| Doubao-Think-On | 406.9 | 294.7 | TPS 虚高 5.6 倍 |
| MiniMax | 116.0 | 290.2 | 接近 Doubao |
| Kimi-Off（方舟）| 22.0 | 99.3 | 被限速了 |
| Kimi-On（方舟）| 68.4 | 110.9 | 被限速了 |

实际文本输出速度排序：**原生 Kimi (520) > Doubao ≈ MiniMax (290) > 方舟 Kimi (100)**。

方舟版 Kimi 的 CPS 只有原生版的 **1/5**。原生版 Kimi 反而是所有模型里最快的——比 Doubao 还快 73%。

---

## 吞吐：并发场景下差距更大

模拟 2 路并发请求（编码 copilot 的典型场景——你在写代码，IDE 同时帮你补全和审查）：

| 指标 | Doubao-Default | **Native-Off** | **Native-On** | MiniMax | Kimi-Off（方舟）| Kimi-On（方舟）|
|------|---------------|------------|-----------|---------|----------|---------|
| RPS | **1.13** | 0.79 | 0.16 | 0.18 | 0.42 | 0.04 |
| 平均延迟 | **1.60s** | 2.44s | 12.30s | 8.70s | 4.71s | 34.19s |
| P95 延迟 | **1.76s** | 2.53s | 12.34s | 10.82s | 4.73s | 46.97s |

Doubao 的吞吐 RPS 仍然最高，但原生 Kimi-Think-Off 的 P95 只有 2.53s，非常接近 Doubao 了。方舟版 Kimi-Think-On 的 P95 高达 47s，差距巨大。

---

## max_tokens 上限实测

顺手测了下各家的 max_tokens 上限，文档上不一定写，得实际试：

| 模型 | 8K | 16K | 32K | 64K | 实际上限 |
|------|-----|------|------|------|---------|
| Doubao Code | ✅ | ✅ | ✅ | ✅ | **≥ 64K** |
| Kimi K2.5 | ✅ | ✅ | ✅ | ❌ 400 | **32K** |
| MiniMax M2.5 | ✅ | ✅ | ✅ | ✅ | **≥ 64K** |

Kimi 设 65536 直接返回 400。如果你的场景需要长输出（比如生成整个模块），32K 可能不够。

---

## Thinking 模式可控性一览

| 模型 | 默认 | 能关吗 | 怎么控制 |
|------|------|--------|---------|
| Doubao Code | **Off** | 可开启 | `thinking: { type: "enabled" }` |
| Kimi K2.5 | **On** | **可关闭** | `thinking: { type: "disabled" }` |
| MiniMax M2.5 | **On** | 暂不支持 | 服务端默认，无参数 |

---

## 最终推荐

### 综合最佳：原生 Kimi-Think-Off 🏆

**→ api.kimi.com + thinking: disabled**

综合 86.7 (S 级)，质量 87.6，CPS 520 字符/秒，总耗时 1.7 分钟。质量和速度双冠军。TTFT 711ms 比 Doubao 略慢但完全可接受。

### 日常编码（copilot / 代码补全 / 快速迭代）

**→ 原生 Kimi-Think-Off** 或 **Doubao-Default**

两者都很快。原生 Kimi 质量更高（87.6 vs 81.2），CPS 更快（520 vs 300）。Doubao 的 TTFT 更低（327ms vs 711ms），如果你对首字延迟极度敏感选 Doubao。

### 高质量场景（code review / 架构设计 / 复杂 bug 修复）

**→ Doubao-Think-On**

质量 89.0，是所有配置中质量最高的。难题提升最大（跳表 95.4）。代价是 6 倍耗时，但你本来也不是在等实时补全。

### Kimi K2.5

**→ 务必用原生 API (api.kimi.com)，务必关闭 thinking。**

- 原生 vs 方舟：CPS 快 5 倍，质量高 5 分
- Think-Off vs Think-On：质量更高，速度快 4 倍

方舟版 Kimi 在这次测试中表现最差（排第 6、7），但换成原生 API 直接登顶。**平台选择比模型选择更重要**。

### MiniMax M2.5

质量 82.1 不错，但被 thinking 拖慢了。等官方开放关闭参数后值得重新测一轮。

---

## 工具开源

项目地址：**[github.com/fakechris/codingplan_benchmark](https://github.com/fakechris/codingplan_benchmark)**

```bash
pip install -r requirements.txt
cp config.example.yaml config.yaml
# 填入 API Key
python -m llm_bench run -c config.yaml
```

支持 OpenAI / Anthropic 兼容协议，基本上市面上的模型都能接。配置文件里每个模型可以单独设 `thinking`、`max_tokens`、`temperature`。

如果你也在选编码模型，欢迎跑一把自己的数据。有问题开 issue。

---

*测试环境：macOS, Python 3.9.6, 中国大陆网络, 2026-02-15 ~ 02-16*
*测试方法：4 道 Medium/Hard/Expert 编码任务, 并发 2, anti-cache 开启*
*所有模型走 Anthropic 兼容协议, 方舟 5 模型同一时间段串行测完, 原生 Kimi 2 模型次日补测并合并*
