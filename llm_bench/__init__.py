"""
LLM Coding Plan Benchmark - 云端 LLM Coding Plan 性能评测工具

评测维度:
  - 速度: TTFT (首 token 延迟), TPS (token/秒), 总延迟
  - 吞吐: 并发请求能力, 限速检测
  - 质量: Coding Plan 结构性、完整性、正确性
  - 一致性: 多次运行结果稳定性 (检测量化/掺水)
"""

__version__ = "0.1.0"
