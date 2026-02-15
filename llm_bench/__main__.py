"""
CLI 入口: python -m llm_bench
"""

from __future__ import annotations

import asyncio
import sys
import click
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()


def _make_progress():
    return Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
        console=console,
    )


def _make_progress_callback(progress, task_progress):
    """创建 progress callback 闭包"""
    def on_progress(phase, current, total, detail):
        if phase not in task_progress:
            task_progress[phase] = progress.add_task(f"[cyan]{phase}[/cyan] {detail}", total=total)
        else:
            progress.update(task_progress[phase], completed=current, description=f"[cyan]{phase}[/cyan] {detail}")
    return on_progress


def _create_provider_from_config(mc):
    """从 ModelConfig 创建 Provider"""
    from .providers import get_provider
    return get_provider(
        provider_name=mc.provider, model=mc.model,
        api_key=mc.get_api_key(), base_url=mc.base_url, timeout=mc.timeout,
        thinking=mc.thinking,
    )


@click.group()
@click.version_option(version="0.2.0")
def cli():
    """LLM Coding Benchmark - 云端 LLM 编码能力评测工具

    \b
    评测维度: 质量 | 速度 | 吞吐 | 一致性
    任务类型: 代码生成 | Bug修复 | 重构 | 测试编写 | 代码审查 | 算法

    \b
    快速开始:
        python -m llm_bench init                          # 生成配置文件
        python -m llm_bench debug -c config.yaml          # 快速连通性测试
        python -m llm_bench run -c config.yaml            # 运行完整评测
        python -m llm_bench run -c config.yaml -d easy    # 只测 easy 难度
        python -m llm_bench tasks                         # 列出所有任务
    """
    pass


# ================================================================
# init - 生成配置
# ================================================================
@cli.command()
@click.option("--output", "-o", default="config.yaml", help="输出配置文件路径")
def init(output):
    """生成示例配置文件"""
    from .config import create_default_config
    path = Path(output)
    if path.exists():
        if not click.confirm(f"{path} 已存在, 是否覆盖?"):
            return
    path.write_text(create_default_config(), encoding="utf-8")
    console.print(f"[green]配置文件已生成: {path}[/green]")
    console.print("请编辑配置文件, 填入你的 API key 和要测试的模型")


# ================================================================
# tasks - 列出任务
# ================================================================
@cli.command()
def tasks():
    """列出所有评测任务"""
    from rich.table import Table
    from rich import box
    from .tasks.coding_plans import TASKS

    table = Table(title="评测任务列表", box=box.ROUNDED)
    table.add_column("ID", style="cyan", width=4)
    table.add_column("任务名", width=28)
    table.add_column("难度", width=8)
    table.add_column("类别", width=16)
    table.add_column("陷阱", width=12)
    table.add_column("标签", width=30)

    dc = {"easy": "green", "medium": "yellow", "hard": "orange3", "expert": "red"}
    for t in TASKS:
        color = dc.get(t.difficulty.value, "white")
        table.add_row(
            t.id, t.title,
            f"[{color}]{t.difficulty.value}[/{color}]",
            t.category.value,
            f"[red]{t.trap_type}[/red]" if t.trap_type else "-",
            ", ".join(t.tags)[:30],
        )
    console.print(table)
    console.print(f"\n共 [bold]{len(TASKS)}[/bold] 个任务 (其中 {len([t for t in TASKS if 'debug' in t.tags])} 个 debug 专用)")


# ================================================================
# providers - 列出 Provider
# ================================================================
@cli.command()
def providers():
    """列出所有支持的 Provider 预设"""
    from rich.table import Table
    from rich import box
    from .providers.registry import PROVIDER_PRESETS

    table = Table(title="支持的 Provider", box=box.ROUNDED)
    table.add_column("名称", style="cyan", width=12)
    table.add_column("类型", width=12)
    table.add_column("Base URL", width=50)
    table.add_column("环境变量", style="dim", width=20)

    for name, preset in PROVIDER_PRESETS.items():
        table.add_row(name, preset.get("class", ""), preset.get("base_url", "-"), preset.get("env_key", ""))
    console.print(table)
    console.print("\n[dim]也可以使用 'openai_compat' / 'anthropic_compat' + base_url 接入任何兼容服务[/dim]")


# ================================================================
# debug - 快速连通性测试
# ================================================================
@cli.command()
@click.option("--config", "-c", required=True, help="配置文件路径")
def debug(config):
    """快速连通性测试 - 每个模型只跑 1 道简单题, 验证 API 是否正常

    \b
    用途: 首次配置后先跑 debug, 确认所有模型都能跑通再做完整评测
    速度: 每个模型 ~5 秒
    """
    asyncio.run(_run_debug(config))


async def _run_debug(config_path):
    from .config import load_config
    from .runner import BenchmarkRunner
    from rich.table import Table
    from rich import box

    cfg = load_config(config_path)
    if not cfg.models:
        console.print("[red]配置文件中没有模型[/red]")
        return

    console.print(f"\n[bold]Debug 模式: 测试 {len(cfg.models)} 个模型的连通性[/bold]\n")

    table = Table(title="连通性测试结果", box=box.ROUNDED)
    table.add_column("模型", width=25, style="bold")
    table.add_column("状态", width=8, justify="center")
    table.add_column("TTFT", width=8, justify="right")
    table.add_column("TPS", width=8, justify="right")
    table.add_column("延迟", width=8, justify="right")
    table.add_column("质量分", width=8, justify="right")
    table.add_column("输出预览", width=40)

    for mc in cfg.models:
        console.print(f"  测试 [cyan]{mc.name}[/cyan] ({mc.model})...", end=" ")
        try:
            provider = _create_provider_from_config(mc)
            runner = BenchmarkRunner(
                provider=provider, model_config=mc,
                global_temperature=cfg.temperature, enable_judge=False,
                task_filter=["D01"],
            )
            tr = await runner.run_debug_task()
            await provider.close()

            if tr.completion.error:
                console.print("[red]FAIL[/red]")
                table.add_row(mc.name, "[red]FAIL[/red]", "-", "-", "-", "-", f"[red]{tr.completion.error[:40]}[/red]")
            else:
                console.print("[green]OK[/green]")
                preview = tr.completion.text.replace("\n", " ")[:40]
                table.add_row(
                    mc.name, "[green]OK[/green]",
                    f"{tr.completion.ttft:.2f}s", f"{tr.completion.tps:.0f}",
                    f"{tr.completion.total_latency:.1f}s",
                    f"{tr.quality.final_score:.0f}",
                    preview,
                )
        except Exception as e:
            console.print("[red]ERROR[/red]")
            table.add_row(mc.name, "[red]ERR[/red]", "-", "-", "-", "-", f"[red]{str(e)[:40]}[/red]")

    console.print()
    console.print(table)
    console.print("\n[dim]全部 OK 后, 运行 'python -m llm_bench run -c config.yaml' 进行完整评测[/dim]")


# ================================================================
# run - 完整评测
# ================================================================
@cli.command()
@click.option("--config", "-c", required=True, help="配置文件路径")
@click.option("--output", "-o", default=None, help="结果输出目录")
@click.option("--no-judge", is_flag=True, help="禁用 Judge 评分")
@click.option("--tasks", "-t", multiple=True, help="指定任务 ID")
@click.option("--difficulty", "-d", type=click.Choice(["easy", "medium", "hard", "expert"]), help="过滤难度")
@click.option("--models", "-m", multiple=True, help="指定模型名称 (可多次使用, 不指定则测全部)")
def run(config, output, no_judge, tasks, difficulty, models):
    """运行完整评测 (支持多模型同时对比)"""
    from .config import load_config

    cfg = load_config(config)
    if output:
        cfg.output_dir = output
    if no_judge:
        cfg.enable_judge = False
    if tasks:
        cfg.task_ids = list(tasks)
    if difficulty:
        cfg.difficulty = difficulty

    # 按名称过滤模型
    if models:
        model_names = set(models)
        cfg.models = [m for m in cfg.models if m.name in model_names]

    if not cfg.models:
        console.print("[red]错误: 没有要测试的模型[/red]")
        sys.exit(1)

    asyncio.run(_run_benchmark(cfg))


async def _run_benchmark(cfg):
    from .config import BenchmarkConfig
    from .runner import BenchmarkRunner, BenchmarkResult
    from .report import print_result, print_comparison, export_json
    from .tasks.coding_plans import TaskDifficulty

    # 创建 Judge provider
    judge_provider = None
    if cfg.enable_judge and cfg.judge_model:
        try:
            judge_provider = _create_provider_from_config(cfg.judge_model)
            console.print(f"[dim]Judge: {cfg.judge_model.model}[/dim]")
        except Exception as e:
            console.print(f"[yellow]Judge 初始化失败, 仅用规则评分: {e}[/yellow]")

    difficulty_filter = TaskDifficulty(cfg.difficulty) if cfg.difficulty else None
    all_results = []

    console.print(f"\n[bold]评测 {len(cfg.models)} 个模型[/bold]")
    console.print(f"[dim]任务过滤: difficulty={cfg.difficulty or 'all'}, tasks={cfg.task_ids or 'all'}[/dim]\n")

    for idx, mc in enumerate(cfg.models):
        console.print(f"\n[bold blue]{'='*60}[/bold blue]")
        console.print(f"[bold blue] [{idx+1}/{len(cfg.models)}] {mc.name} ({mc.model})[/bold blue]")
        console.print(f"[bold blue]{'='*60}[/bold blue]")

        try:
            provider = _create_provider_from_config(mc)
        except Exception as e:
            console.print(f"[red]初始化失败: {e}[/red]")
            continue

        with _make_progress() as progress:
            task_progress = {}
            runner = BenchmarkRunner(
                provider=provider, model_config=mc, judge_provider=judge_provider,
                global_temperature=cfg.temperature, concurrency=cfg.concurrency,
                consistency_runs=cfg.consistency_runs, enable_judge=cfg.enable_judge,
                task_filter=cfg.task_ids, difficulty_filter=difficulty_filter,
                progress_callback=_make_progress_callback(progress, task_progress),
                throughput_multiplier=cfg.throughput_multiplier,
                max_quality_tasks=cfg.max_quality_tasks,
                anti_cache=cfg.anti_cache,
            )
            result = await runner.run_full_benchmark()
            all_results.append(result)
            print_result(result)

        await provider.close()

    if judge_provider:
        await judge_provider.close()

    # === 多模型对比 ===
    if len(all_results) > 1:
        print_comparison(all_results)

    # === 导出 ===
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if "json" in cfg.output_format:
        export_json(all_results, str(output_dir / "benchmark_results.json"))


# ================================================================
# quick - 快速测试单个模型 (无需配置文件)
# ================================================================
@cli.command()
@click.option("--provider", "-p", required=True, help="Provider 名称")
@click.option("--model", "-m", required=True, help="模型名称")
@click.option("--api-key", "-k", default=None, help="API Key")
@click.option("--base-url", default=None, help="自定义 base URL")
@click.option("--tasks", "-t", multiple=True, help="指定任务 ID")
@click.option("--difficulty", "-d", type=click.Choice(["easy", "medium", "hard", "expert"]), help="过滤难度")
@click.option("--no-judge", is_flag=True, help="禁用 Judge 评分")
@click.option("--no-throughput", is_flag=True, help="跳过吞吐测试")
@click.option("--no-consistency", is_flag=True, help="跳过一致性测试")
@click.option("--temperature", type=float, default=0.0, help="温度")
@click.option("--max-tokens", type=int, default=None, help="最大输出 token")
def quick(provider, model, api_key, base_url, tasks, difficulty, no_judge, no_throughput, no_consistency, temperature, max_tokens):
    """快速测试单个模型 (无需配置文件)"""
    from .config import ModelConfig
    mc = ModelConfig(
        name=model, provider=provider, model=model,
        api_key=api_key, base_url=base_url,
        temperature=temperature if temperature != 0.0 else None,
        max_tokens=max_tokens,
    )
    asyncio.run(_quick_run(mc, tasks, difficulty, no_judge, no_throughput, no_consistency))


async def _quick_run(mc, task_ids, difficulty, no_judge, no_throughput, no_consistency):
    from .providers import get_provider
    from .runner import BenchmarkRunner, BenchmarkResult, TaskResult
    from .providers.base import CompletionResult
    from .scorer import QualityScore
    from .report import print_result
    from .tasks.coding_plans import TaskDifficulty
    from datetime import datetime

    try:
        provider = _create_provider_from_config(mc)
    except Exception as e:
        console.print(f"[red]初始化失败: {e}[/red]")
        return

    difficulty_filter = TaskDifficulty(difficulty) if difficulty else None

    with _make_progress() as progress:
        task_progress = {}
        runner = BenchmarkRunner(
            provider=provider, model_config=mc, global_temperature=0.0,
            enable_judge=not no_judge,
            task_filter=list(task_ids) if task_ids else None,
            difficulty_filter=difficulty_filter,
            progress_callback=_make_progress_callback(progress, task_progress),
        )

        result = BenchmarkResult(
            model=mc.model, provider=mc.provider,
            timestamp=datetime.now().isoformat(), model_config_name=mc.name,
        )

        tasks_to_run = runner._get_tasks()
        runner._report("quality", 0, len(tasks_to_run), "开始...")
        for i, task in enumerate(tasks_to_run):
            runner._report("quality", i+1, len(tasks_to_run), f"[{task.id}] {task.title}")
            try:
                tr = await runner.run_single_task(task)
                result.task_results.append(tr)
            except Exception as e:
                comp = CompletionResult(text="", model=mc.model, error=str(e))
                result.task_results.append(TaskResult(task=task, completion=comp, quality=QualityScore(task_id=task.id)))

        if not no_throughput:
            runner._report("throughput", 0, 1, "吞吐测试...")
            try:
                result.throughput_result = await runner.run_throughput_test()
                runner._report("throughput", 1, 1, "完成")
            except Exception as e:
                runner._report("throughput", 1, 1, f"失败: {e}")

        if not no_consistency:
            runner._report("consistency", 0, 1, "一致性测试...")
            try:
                result.consistency_results = await runner.run_consistency_test()
                runner._report("consistency", 1, 1, "完成")
            except Exception as e:
                runner._report("consistency", 1, 1, f"失败: {e}")

        result.compute_overall()
        print_result(result)

    await provider.close()


# ================================================================
# compare - 对比历史结果
# ================================================================
@cli.command()
@click.argument("json_files", nargs=-1, required=True)
def compare(json_files):
    """对比多个结果 JSON 文件"""
    import json
    from .runner import BenchmarkResult
    from .report import print_comparison

    all_results = []
    for f in json_files:
        path = Path(f)
        if not path.exists():
            console.print(f"[red]文件不存在: {f}[/red]")
            continue
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        for r in data.get("results", []):
            br = BenchmarkResult(
                model=r["model"], provider=r["provider"], timestamp=r["timestamp"],
                model_config_name=r.get("model_name", r["model"]),
            )
            scores = r.get("scores", {})
            br.overall_score = scores.get("overall", 0)
            br.quality_score = scores.get("quality", 0)
            br.speed_score = scores.get("speed", 0)
            br.throughput_score = scores.get("throughput", 0)
            br.consistency_score = scores.get("consistency", 0)
            all_results.append(br)

    if all_results:
        print_comparison(all_results)
    else:
        console.print("[red]没有找到有效数据[/red]")


if __name__ == "__main__":
    cli()
