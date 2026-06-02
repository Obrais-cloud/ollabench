"""ollabench CLI."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import anyio
import click
from rich.console import Console
from rich.table import Table

from . import __version__
from .compare import diff, filter_significant
from .config import (
    BenchPlan,
    TargetSpec,
    expand_targets,
    find_fleet_config,
    load_bench,
    load_fleet,
)
from .report import render_markdown, render_terminal, write_bundle, serialize_json
from .runner import CellRun, run_plan
from .scenarios import SCENARIOS, resolve_scenarios


console = Console()


def _progress(cell: CellRun) -> None:
    label = f"{cell.model} @ {cell.node} / {cell.scenario} c={cell.concurrency}"
    if not cell.available:
        console.print(f"[dim]  {label}: {cell.error}[/dim]")
        return
    if cell.error:
        console.print(f"[red]  {label}: {cell.error}[/red]")
        return
    if cell.ttft is None:
        console.print(f"[yellow]  {label}: no successful samples[/yellow]")
        return
    console.print(
        f"  [green]✓[/green] {label}: "
        f"ttft p50={cell.ttft.p50:.0f}ms  "
        f"gen={cell.gen_tok_s.p50:.1f}t/s  "
        f"total p50={cell.total.p50:.0f}ms"
        + (f"  thr={cell.throughput_tok_s:.1f}t/s" if cell.concurrency > 1 else "")
    )


@click.group()
@click.version_option(__version__, prog_name="ollabench")
def cli() -> None:
    """Performance benchmark matrix for an Ollama fleet."""


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--fleet", "fleet_path", type=click.Path(), default=None,
              help="Path to fleet.yml (ollafleet schema). Auto-discovered if omitted.")
@click.option("--bench", "bench_path", type=click.Path(), default=None,
              help="Path to bench.yml. If omitted, --model and --scenario are required.")
@click.option("--model", "models", multiple=True,
              help="Model spec 'name' or 'name@node'. Repeatable.")
@click.option("--scenario", "scenario_names", multiple=True,
              help="Scenario name. Repeatable. Defaults to short,medium,long_gen.")
@click.option("--iterations", "-n", type=int, default=None,
              help="Timed samples per cell (after warmup).")
@click.option("--warmup", type=int, default=None, help="Warmup runs per cell.")
@click.option("--concurrency", "-c", type=int, multiple=True,
              help="Concurrency level. Repeatable to sweep, e.g. -c 1 -c 4.")
@click.option("--out", "out_dir", type=click.Path(), default="./bench-out",
              help="Output directory.")
def run(
    fleet_path: str | None,
    bench_path: str | None,
    models: tuple[str, ...],
    scenario_names: tuple[str, ...],
    iterations: int | None,
    warmup: int | None,
    concurrency: tuple[int, ...],
    out_dir: str,
) -> None:
    """Run the benchmark plan and write results to OUT_DIR."""

    if bench_path:
        plan = load_bench(Path(bench_path))
    else:
        if not models:
            raise click.UsageError(
                "Either --bench bench.yml or one or more --model are required."
            )
        targets: list[TargetSpec] = []
        for m in models:
            if "@" in m:
                model, node = m.split("@", 1)
                targets.append(TargetSpec(model=model, node=node))
            else:
                targets.append(TargetSpec(model=m))
        scenarios_default = list(scenario_names) or ["short", "medium", "long_gen"]
        plan = BenchPlan(
            targets=targets,
            scenarios=scenarios_default,
            iterations=iterations or 5,
            warmup=warmup if warmup is not None else 1,
            concurrency=list(concurrency) or [1],
        )

    # CLI flags override bench.yml when both present.
    if scenario_names:
        plan.scenarios = list(scenario_names)
    if iterations is not None:
        plan.iterations = iterations
    if warmup is not None:
        plan.warmup = warmup
    if concurrency:
        plan.concurrency = list(concurrency)

    scenarios = resolve_scenarios(plan.scenarios)
    nodes = load_fleet(find_fleet_config(fleet_path))
    pairs = expand_targets(plan.targets, nodes)
    if not pairs:
        raise click.UsageError("No (model, node) pairs to run.")

    console.print(
        f"[bold]ollabench[/bold] — {len(pairs)} cells × "
        f"{len(scenarios)} scenarios × {len(plan.concurrency)} conc-levels"
    )
    for model, node in pairs:
        console.print(f"  - {model} @ {node.name} ({node.host})")
    console.print(
        f"  iterations={plan.iterations} warmup={plan.warmup} "
        f"concurrency={plan.concurrency}"
    )
    console.print("")

    results = anyio.run(run_plan, pairs, scenarios, plan, _progress)

    meta = {
        "timestamp": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "iterations": plan.iterations,
        "warmup": plan.warmup,
        "scenarios": plan.scenarios,
        "concurrency": plan.concurrency,
        "ollabench_version": __version__,
    }

    console.print("")
    render_terminal(results, console)

    json_path, md_path = write_bundle(Path(out_dir), results, meta)
    console.print(f"\n[green]wrote[/green] {json_path}")
    console.print(f"[green]wrote[/green] {md_path}")
    console.print(f"[green]wrote[/green] {Path(out_dir) / 'bench-latest.json'}  (for ollarouter)")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--format", "fmt",
              type=click.Choice(["terminal", "markdown", "json"]),
              default="terminal")
def report(path: str, fmt: str) -> None:
    """Re-render a bench-*.json result file."""
    data = json.loads(Path(path).read_text())
    meta = data.get("meta", {})
    cells = [_cell_from_dict(r) for r in data.get("results", [])]

    if fmt == "terminal":
        render_terminal(cells, console)
    elif fmt == "markdown":
        click.echo(render_markdown(cells, meta))
    else:
        click.echo(json.dumps(serialize_json(cells, meta), indent=2))


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("a", type=click.Path(exists=True))
@click.argument("b", type=click.Path(exists=True))
@click.option("--threshold", type=float, default=10.0,
              help="Only show deltas with |pct| >= threshold.")
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
def compare(a: str, b: str, threshold: float, as_json: bool) -> None:
    """Compare two bench JSONs: A baseline vs B new."""
    deltas = diff(Path(a), Path(b))
    sig = filter_significant(deltas, threshold)

    if as_json:
        click.echo(json.dumps(
            [
                {
                    "key": d.key,
                    "metric": d.metric,
                    "a": d.a,
                    "b": d.b,
                    "pct": round(d.pct, 2),
                    "regression": d.regressed,
                }
                for d in sig
            ],
            indent=2,
        ))
        return

    if not sig:
        console.print(
            f"[green]no significant changes[/green] (|pct| < {threshold:.1f}%)"
        )
        return

    table = Table(title=f"deltas vs baseline (|pct| ≥ {threshold:.1f}%)", header_style="bold")
    table.add_column("cell")
    table.add_column("metric")
    table.add_column("baseline", justify="right")
    table.add_column("new", justify="right")
    table.add_column("Δ%", justify="right")
    table.add_column("verdict")

    sig.sort(key=lambda d: -abs(d.pct))
    for d in sig:
        verdict = "[red]regressed[/red]" if d.regressed else "[green]improved[/green]"
        table.add_row(
            d.key,
            d.metric,
            f"{d.a:.2f}",
            f"{d.b:.2f}",
            f"{d.pct:+.1f}%",
            verdict,
        )
    console.print(table)


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------

@cli.command(name="scenarios")
def list_scenarios() -> None:
    """List built-in scenarios."""
    table = Table(header_style="bold")
    table.add_column("name")
    table.add_column("num_predict", justify="right")
    table.add_column("description")
    for name, s in SCENARIOS.items():
        table.add_row(name, str(s.num_predict), s.description)
    console.print(table)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _cell_from_dict(d: dict) -> CellRun:
    """Hydrate a CellRun (sans raw samples) for re-rendering."""
    from .stats import Summary

    def _sum(x: dict | None) -> Summary | None:
        if x is None:
            return None
        return Summary(
            n=int(x["n"]),
            mean=float(x["mean"]),
            p50=float(x["p50"]),
            p95=float(x["p95"]),
            p99=float(x["p99"]),
            min=float(x["min"]),
            max=float(x["max"]),
        )

    cell = CellRun(
        model=d["model"],
        node=d["node"],
        scenario=d["scenario"],
        concurrency=int(d.get("concurrency", 1)),
        available=bool(d.get("available", True)),
        error=d.get("error"),
    )
    cell.successes = int(d.get("successes", 0))
    cell.failures = int(d.get("failures", 0))
    cell.wall_run_ms = float(d.get("wall_run_ms", 0))
    cell.throughput_req_s = float(d.get("throughput_req_s", 0))
    cell.throughput_tok_s = float(d.get("throughput_tok_s", 0))
    cell.ttft = _sum(d.get("ttft_ms"))
    cell.total = _sum(d.get("total_ms"))
    cell.gen_tok_s = _sum(d.get("gen_tok_s"))
    cell.wall_tok_s = _sum(d.get("wall_tok_s"))
    cell.eval_count = _sum(d.get("eval_count"))
    cell.prompt_eval_count = _sum(d.get("prompt_eval_count"))
    return cell


if __name__ == "__main__":
    cli()
