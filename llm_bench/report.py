"""
报告生成器

支持:
1. 终端彩色输出 (Rich)
2. JSON 导出
3. 多模型对比 - 综合排行 + 客观指标 head-to-head
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from .runner import BenchmarkResult, TaskResult

console = Console()


# ── helpers ──────────────────────────────────────────────────────
def _sc(score: float) -> str:
    """score → color"""
    if score >= 80: return "green"
    if score >= 60: return "yellow"
    if score >= 40: return "orange3"
    return "red"


def _grade(score: float) -> str:
    if score >= 90: return "S"
    if score >= 80: return "A"
    if score >= 70: return "B"
    if score >= 60: return "C"
    if score >= 40: return "D"
    return "F"


def _label(r: BenchmarkResult) -> str:
    return r.model_config_name or r.model


def _fmt_time(s: float) -> str:
    """秒 → 人类友好"""
    if s <= 0: return "-"
    if s < 0.01: return f"{s*1000:.1f}ms"
    if s < 1: return f"{s*1000:.0f}ms"
    if s < 60: return f"{s:.2f}s"
    return f"{s/60:.1f}min"


def _fmt_tps(v: float) -> str:
    if v <= 0: return "-"
    return f"{v:.1f}"


def _fmt_int(v: int) -> str:
    if v <= 0: return "-"
    if v >= 10000: return f"{v/1000:.1f}k"
    return str(v)


def _best_marker(values: list[float], idx: int, higher_better: bool = True) -> str:
    """判断是否是最佳值"""
    if not values:
        return ""
    target = max(values) if higher_better else min(v for v in values if v > 0)
    return " ★" if values[idx] == target and target > 0 else ""


# ── 单模型报告 ───────────────────────────────────────────────────
def print_result(result: BenchmarkResult):
    """打印单个模型的完整测试结果"""
    console.print()
    label = _label(result)
    console.print(Panel(f"  Benchmark Report: {label}  ", style="bold blue", expand=False))
    console.print()

    # ── 综合评分 ──
    st = Table(title="综合评分", box=box.ROUNDED, show_header=True)
    st.add_column("维度", style="bold")
    st.add_column("得分", justify="right")
    st.add_column("等级", justify="center")
    st.add_column("权重", justify="right", style="dim")

    for name, score, w in [
        ("质量 Quality", result.quality_score, "45%"),
        ("速度 Speed", result.speed_score, "25%"),
        ("吞吐 Throughput", result.throughput_score, "15%"),
        ("一致性 Consistency", result.consistency_score, "15%"),
    ]:
        c = _sc(score)
        st.add_row(name, f"[{c}]{score:.1f}[/{c}]", f"[{c}]{_grade(score)}[/{c}]", w)

    oc = _sc(result.overall_score)
    st.add_row("[bold]综合 Overall[/bold]",
               f"[bold {oc}]{result.overall_score:.1f}[/bold {oc}]",
               f"[bold {oc}]{_grade(result.overall_score)}[/bold {oc}]", "100%", style="bold")
    console.print(st)
    console.print()

    # ── 客观指标汇总 ──
    _print_objective_summary(result)

    # ── 逐题客观指标 ──
    _print_per_task_metrics(result)

    # ── 吞吐测试 ──
    if result.throughput_result:
        _print_throughput(result)

    # ── 一致性测试 ──
    if result.consistency_results:
        _print_consistency(result)


def _print_objective_summary(result: BenchmarkResult):
    """客观指标汇总表"""
    m = result.get_objective_metrics()
    if not m:
        return

    table = Table(title="客观指标汇总 (Raw Metrics)", box=box.ROUNDED, show_header=True)
    table.add_column("指标", style="bold", width=26)
    table.add_column("值", justify="right", width=18)
    table.add_column("", width=4)
    table.add_column("指标", style="bold", width=26)
    table.add_column("值", justify="right", width=18)

    # 左右两列排布
    rows = [
        ("测试总耗时 (Wall Time)", _fmt_time(m["wall_time"]),
         "成功率", f"{m['success_rate']:.0f}% ({m['task_count']} 题)"),
        ("平均 TTFT (首字延迟)", _fmt_time(m["ttft"]["avg"]),
         "TTFT 范围", f"{_fmt_time(m['ttft']['min'])} ~ {_fmt_time(m['ttft']['max'])}"),
        ("平均 TPS (token/秒)", _fmt_tps(m["tps"]["avg"]),
         "TPS 范围", f"{_fmt_tps(m['tps']['min'])} ~ {_fmt_tps(m['tps']['max'])}"),
        ("平均 CPS (字符/秒)", _fmt_tps(m["cps"]["avg"]),
         "CPS 范围", f"{_fmt_tps(m['cps']['min'])} ~ {_fmt_tps(m['cps']['max'])}"),
        ("平均生成时间 (Gen Time)", _fmt_time(m["gen_time"]["avg"]),
         "平均总延迟 (Latency)", _fmt_time(m["latency"]["avg"])),
        ("总输出 Tokens", _fmt_int(m["total_output_tokens"]),
         "总输出字符", _fmt_int(m["total_output_chars"])),
    ]

    for left_k, left_v, right_k, right_v in rows:
        table.add_row(left_k, left_v, "│", right_k, right_v)

    console.print(table)
    console.print()


def _print_per_task_metrics(result: BenchmarkResult):
    """逐题客观指标详情表"""
    if not result.task_results:
        return

    tt = Table(title="逐题详情 (Per-Task Metrics)", box=box.SIMPLE_HEAVY, show_header=True)
    tt.add_column("ID", style="cyan", width=4)
    tt.add_column("任务", width=22)
    tt.add_column("TTFT", justify="right", width=8)
    tt.add_column("生成", justify="right", width=8)
    tt.add_column("总延迟", justify="right", width=8)
    tt.add_column("TPS", justify="right", width=7)
    tt.add_column("CPS", justify="right", width=7)
    tt.add_column("Tokens", justify="right", width=7)
    tt.add_column("字符", justify="right", width=7)
    tt.add_column("质量", justify="right", width=6)
    tt.add_column("", width=4)

    for tr in result.task_results:
        c = tr.completion
        status = "[red]FAIL[/red]" if c.error else "[green]OK[/green]"
        qc = _sc(tr.quality.final_score)
        tt.add_row(
            tr.task.id,
            tr.task.title[:22],
            _fmt_time(c.ttft),
            _fmt_time(c.gen_time),
            _fmt_time(c.total_latency),
            _fmt_tps(c.tps),
            _fmt_tps(c.chars_per_second),
            _fmt_int(c.completion_tokens),
            _fmt_int(c.output_chars),
            f"[{qc}]{tr.quality.final_score:.1f}[/{qc}]",
            status,
        )
    console.print(tt)
    console.print()


def _print_throughput(result: BenchmarkResult):
    """吞吐测试结果"""
    t = result.throughput_result
    tp = Table(title="吞吐测试 (Throughput)", box=box.ROUNDED)
    tp.add_column("指标", style="bold")
    tp.add_column("值", justify="right")
    tp.add_row("并发数", str(t.concurrency))
    tp.add_row("成功/总数", f"[green]{t.successful}[/green]/{t.total_requests}")
    tp.add_row("RPS (请求/秒)", f"{t.requests_per_second:.2f}")
    tp.add_row("总耗时", _fmt_time(t.total_time))
    tp.add_row("平均延迟", _fmt_time(t.avg_latency))
    tp.add_row("P50 延迟", _fmt_time(t.p50_latency))
    tp.add_row("P95 延迟", _fmt_time(t.p95_latency))
    tp.add_row("P99 延迟", _fmt_time(t.p99_latency))
    tp.add_row("平均 TPS", f"{t.avg_tps:.1f} tok/s")
    if t.errors:
        tp.add_row("首个错误", f"[red]{t.errors[0][:60]}[/red]")
    console.print(tp)
    console.print()


def _print_consistency(result: BenchmarkResult):
    """一致性测试结果"""
    ct = Table(title="一致性测试 (量化/掺水检测)", box=box.ROUNDED)
    ct.add_column("任务", style="bold")
    ct.add_column("次数", justify="right")
    ct.add_column("平均分", justify="right")
    ct.add_column("标准差", justify="right")
    ct.add_column("相似度", justify="right")
    ct.add_column("判定", justify="center")
    for cr in result.consistency_results:
        if cr.score_stddev > 15: verdict = "[red]可疑[/red]"
        elif cr.score_stddev > 8: verdict = "[yellow]一般[/yellow]"
        else: verdict = "[green]稳定[/green]"
        ct.add_row(cr.task_id, str(cr.num_runs), f"{cr.avg_score:.1f}",
                   f"{cr.score_stddev:.2f}", f"{cr.responses_similarity:.2f}", verdict)
    console.print(ct)
    console.print()


# ── 多模型对比 ───────────────────────────────────────────────────
def print_comparison(results: list[BenchmarkResult]):
    """打印多模型对比报告 — 客观指标为核心"""
    if not results:
        return

    console.print()
    console.print(Panel("  模型对比排行榜  ", style="bold magenta", expand=False))
    console.print()

    results_sorted = sorted(results, key=lambda r: r.overall_score, reverse=True)

    # ── 1. 综合评分排行 ──
    _print_rank_table(results_sorted)

    # ── 2. 客观指标横向对比 (核心表) ──
    _print_metrics_comparison(results_sorted)

    # ── 3. 逐题 TTFT / TPS / 延迟 / 质量 对比 ──
    _print_task_comparison(results_sorted)

    # ── 4. 吞吐对比 ──
    _print_throughput_comparison(results_sorted)


def _print_rank_table(results: list[BenchmarkResult]):
    """综合评分排行表"""
    rt = Table(title="综合评分排行", box=box.DOUBLE_EDGE, show_header=True)
    rt.add_column("#", width=3, justify="right")
    rt.add_column("模型", width=25, style="bold")
    rt.add_column("综合", justify="right", width=6)
    rt.add_column("质量", justify="right", width=6)
    rt.add_column("速度", justify="right", width=6)
    rt.add_column("吞吐", justify="right", width=6)
    rt.add_column("一致性", justify="right", width=6)
    rt.add_column("等级", justify="center", width=4)

    for i, r in enumerate(results):
        oc = _sc(r.overall_score)
        rt.add_row(
            str(i + 1), _label(r),
            f"[{oc}]{r.overall_score:.1f}[/{oc}]",
            f"[{_sc(r.quality_score)}]{r.quality_score:.1f}[/{_sc(r.quality_score)}]",
            f"[{_sc(r.speed_score)}]{r.speed_score:.1f}[/{_sc(r.speed_score)}]",
            f"[{_sc(r.throughput_score)}]{r.throughput_score:.1f}[/{_sc(r.throughput_score)}]",
            f"[{_sc(r.consistency_score)}]{r.consistency_score:.1f}[/{_sc(r.consistency_score)}]",
            f"[{oc}]{_grade(r.overall_score)}[/{oc}]",
        )
    console.print(rt)
    console.print()


def _print_metrics_comparison(results: list[BenchmarkResult]):
    """客观指标横向对比 — 所有 raw numbers"""
    metrics_list = [r.get_objective_metrics() for r in results]
    labels = [_label(r) for r in results]

    table = Table(title="客观指标对比 (Objective Metrics Comparison)", box=box.HEAVY_EDGE, show_header=True)
    table.add_column("指标", style="bold", width=22)
    for lb in labels:
        table.add_column(lb[:18], justify="right", width=18)

    # 定义要展示的行: (显示名, 取值函数, 格式函数, higher_is_better)
    rows_def = [
        ("测试总耗时",       lambda m: m.get("wall_time", 0),                _fmt_time,  False),
        ("成功率",           lambda m: m.get("success_rate", 0),             lambda v: f"{v:.0f}%", True),
        ("─── TTFT (首字延迟) ───", None, None, None),
        ("  平均 TTFT",      lambda m: m.get("ttft", {}).get("avg", 0),     _fmt_time,  False),
        ("  最小 TTFT",      lambda m: m.get("ttft", {}).get("min", 0),     _fmt_time,  False),
        ("  最大 TTFT",      lambda m: m.get("ttft", {}).get("max", 0),     _fmt_time,  False),
        ("  P50 TTFT",       lambda m: m.get("ttft", {}).get("p50", 0),     _fmt_time,  False),
        ("─── TPS (token/秒) ───", None, None, None),
        ("  平均 TPS",       lambda m: m.get("tps", {}).get("avg", 0),      _fmt_tps,   True),
        ("  最小 TPS",       lambda m: m.get("tps", {}).get("min", 0),      _fmt_tps,   True),
        ("  最大 TPS",       lambda m: m.get("tps", {}).get("max", 0),      _fmt_tps,   True),
        ("─── CPS (字符/秒) ───", None, None, None),
        ("  平均 CPS",       lambda m: m.get("cps", {}).get("avg", 0),      _fmt_tps,   True),
        ("  最小 CPS",       lambda m: m.get("cps", {}).get("min", 0),      _fmt_tps,   True),
        ("  最大 CPS",       lambda m: m.get("cps", {}).get("max", 0),      _fmt_tps,   True),
        ("─── 延迟 ───", None, None, None),
        ("  平均生成时间",    lambda m: m.get("gen_time", {}).get("avg", 0), _fmt_time,  False),
        ("  平均总延迟",      lambda m: m.get("latency", {}).get("avg", 0),  _fmt_time,  False),
        ("  最大总延迟",      lambda m: m.get("latency", {}).get("max", 0),  _fmt_time,  False),
        ("─── 输出量 ───", None, None, None),
        ("  总输出 Tokens",  lambda m: m.get("total_output_tokens", 0),     _fmt_int,   None),
        ("  总输出字符",     lambda m: m.get("total_output_chars", 0),      _fmt_int,   None),
    ]

    for row_name, getter, formatter, higher_better in rows_def:
        if getter is None:
            # 分隔行
            sep_row = [f"[dim]{row_name}[/dim]"] + ["" for _ in labels]
            table.add_row(*sep_row)
            continue

        values = [getter(m) for m in metrics_list]
        cells = [row_name]
        for i, v in enumerate(values):
            text = formatter(v) if v else "-"
            # 高亮最佳值
            if higher_better is not None and v and v > 0:
                if higher_better:
                    is_best = v == max(vv for vv in values if vv > 0)
                else:
                    is_best = v == min(vv for vv in values if vv > 0)
                if is_best:
                    text = f"[bold green]{text}[/bold green]"
            cells.append(text)
        table.add_row(*cells)

    console.print(table)
    console.print()


def _print_task_comparison(results: list[BenchmarkResult]):
    """逐题 head-to-head 对比 (TTFT / TPS / 延迟 / 质量)"""
    # 收集所有 task
    all_task_ids = []
    task_titles = {}
    for r in results:
        for tr in r.task_results:
            if tr.task.id not in task_titles:
                all_task_ids.append(tr.task.id)
                task_titles[tr.task.id] = tr.task.title

    if not all_task_ids:
        return

    # task_id -> model_label -> TaskResult
    task_map = {}
    for r in results:
        label = _label(r)
        for tr in r.task_results:
            task_map.setdefault(tr.task.id, {})[label] = tr

    labels = [_label(r) for r in results]

    # 对每个指标生成一个对比表
    for metric_name, metric_fn, fmt_fn, higher_better in [
        ("TTFT (首字延迟)", lambda tr: tr.completion.ttft, _fmt_time, False),
        ("TPS (token/秒)", lambda tr: tr.completion.tps, _fmt_tps, True),
        ("总延迟 (Latency)", lambda tr: tr.completion.total_latency, _fmt_time, False),
        ("质量分 (Quality)", lambda tr: tr.quality.final_score, lambda v: f"{v:.1f}", True),
    ]:
        table = Table(title=f"逐题对比: {metric_name}", box=box.SIMPLE_HEAVY)
        table.add_column("任务", width=22, style="bold")
        for lb in labels:
            table.add_column(lb[:15], justify="right", width=14)

        for tid in all_task_ids:
            row = [f"[cyan]{tid}[/cyan] {task_titles[tid][:15]}"]
            values = []
            for lb in labels:
                tr = task_map.get(tid, {}).get(lb)
                if tr and not tr.completion.error:
                    values.append(metric_fn(tr))
                else:
                    values.append(0)

            # 找最佳
            positive = [v for v in values if v > 0]
            if positive:
                best = max(positive) if higher_better else min(positive)
            else:
                best = None

            for v in values:
                if v <= 0:
                    row.append("[red]FAIL[/red]")
                elif best is not None and v == best:
                    row.append(f"[bold green]{fmt_fn(v)}[/bold green]")
                else:
                    row.append(fmt_fn(v))

            table.add_row(*row)

        console.print(table)
        console.print()


def _print_throughput_comparison(results: list[BenchmarkResult]):
    """吞吐测试对比"""
    has_tp = any(r.throughput_result for r in results)
    if not has_tp:
        return

    labels = [_label(r) for r in results]
    table = Table(title="吞吐测试对比 (Throughput)", box=box.ROUNDED)
    table.add_column("指标", style="bold", width=18)
    for lb in labels:
        table.add_column(lb[:18], justify="right", width=16)

    rows_def = [
        ("成功率", lambda t: f"{t.successful}/{t.total_requests}" if t else "-"),
        ("RPS", lambda t: f"{t.requests_per_second:.2f}" if t else "-"),
        ("平均延迟", lambda t: _fmt_time(t.avg_latency) if t else "-"),
        ("P50 延迟", lambda t: _fmt_time(t.p50_latency) if t else "-"),
        ("P95 延迟", lambda t: _fmt_time(t.p95_latency) if t else "-"),
        ("P99 延迟", lambda t: _fmt_time(t.p99_latency) if t else "-"),
        ("平均 TPS", lambda t: f"{t.avg_tps:.1f} tok/s" if t else "-"),
    ]

    for name, fn in rows_def:
        row = [name]
        for r in results:
            row.append(fn(r.throughput_result))
        table.add_row(*row)

    console.print(table)
    console.print()


# ── JSON 导出 ────────────────────────────────────────────────────
def export_json(results: list[BenchmarkResult], output_path: str):
    data = {
        "benchmark": "llm-coding-bench",
        "version": "0.3.0",
        "results": [r.to_dict() for r in results],
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    console.print(f"[green]结果已导出: {path}[/green]")
