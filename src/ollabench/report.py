"""Rendering: rich tables for the terminal + markdown for files +
a compact JSON matrix for downstream consumers (ollarouter)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .runner import CellRun


def _fmt(v: float | None, suffix: str = "", digits: int = 1) -> str:
    if v is None:
        return "-"
    return f"{v:.{digits}f}{suffix}"


def render_terminal(results: list[CellRun], console: Console) -> None:
    by_scenario: dict[str, list[CellRun]] = {}
    for r in results:
        by_scenario.setdefault(r.scenario, []).append(r)

    for scenario, rows in by_scenario.items():
        rows.sort(
            key=lambda r: (r.concurrency, r.model, r.node),
        )
        table = Table(
            title=f"scenario: {scenario}",
            header_style="bold",
            show_lines=False,
        )
        table.add_column("model")
        table.add_column("node")
        table.add_column("conc", justify="right")
        table.add_column("ttft p50", justify="right")
        table.add_column("ttft p95", justify="right")
        table.add_column("gen tok/s p50", justify="right")
        table.add_column("total p50", justify="right")
        table.add_column("thr req/s", justify="right")
        table.add_column("thr tok/s", justify="right")
        table.add_column("ok/fail", justify="right")

        for r in rows:
            if not r.available:
                table.add_row(
                    r.model, r.node, str(r.concurrency),
                    "-", "-", "-", "-", "-", "-",
                    f"[red]{r.error or 'unavail'}[/red]",
                )
                continue
            table.add_row(
                r.model,
                r.node,
                str(r.concurrency),
                _fmt(r.ttft.p50 if r.ttft else None, "ms", 0),
                _fmt(r.ttft.p95 if r.ttft else None, "ms", 0),
                _fmt(r.gen_tok_s.p50 if r.gen_tok_s else None, "", 1),
                _fmt(r.total.p50 if r.total else None, "ms", 0),
                _fmt(r.throughput_req_s, "", 2) if r.concurrency > 1 else "-",
                _fmt(r.throughput_tok_s, "", 1) if r.concurrency > 1 else "-",
                f"{r.successes}/{r.failures}",
            )
        console.print(table)


def serialize_json(results: list[CellRun], meta: dict) -> dict:
    return {
        "meta": meta,
        "results": [r.as_dict() for r in results],
        "matrix": _build_router_matrix(results),
    }


def _build_router_matrix(results: list[CellRun]) -> dict:
    """Compact (model, node, scenario) -> headline metrics for ollarouter.

    Only concurrency=1 cells are used here — they're the per-request baseline."""

    matrix: dict = {}
    for r in results:
        if r.concurrency != 1 or not r.available or r.ttft is None:
            continue
        key = f"{r.model}|{r.node}|{r.scenario}"
        matrix[key] = {
            "model": r.model,
            "node": r.node,
            "scenario": r.scenario,
            "ttft_p50_ms": round(r.ttft.p50, 2),
            "ttft_p95_ms": round(r.ttft.p95, 2),
            "gen_tok_s_p50": round(r.gen_tok_s.p50, 3) if r.gen_tok_s else 0.0,
            "gen_tok_s_p95": round(r.gen_tok_s.p95, 3) if r.gen_tok_s else 0.0,
            "total_p50_ms": round(r.total.p50, 2) if r.total else 0.0,
            "cold_ttft_ms": (
                round(r.cold_sample.ttft_ms, 2)
                if r.cold_sample and r.cold_sample.ttft_ms is not None
                else None
            ),
        }
    return matrix


def render_markdown(results: list[CellRun], meta: dict) -> str:
    lines: list[str] = []
    lines.append(f"# ollabench report — {meta.get('timestamp', '?')}")
    lines.append("")
    lines.append(f"- iterations: {meta.get('iterations')}")
    lines.append(f"- warmup: {meta.get('warmup')}")
    lines.append(f"- scenarios: {', '.join(meta.get('scenarios', []))}")
    lines.append(f"- concurrency: {meta.get('concurrency')}")
    lines.append("")

    by_scenario: dict[str, list[CellRun]] = {}
    for r in results:
        by_scenario.setdefault(r.scenario, []).append(r)

    for scenario, rows in by_scenario.items():
        rows.sort(key=lambda r: (r.concurrency, r.model, r.node))
        lines.append(f"## scenario: `{scenario}`")
        lines.append("")
        lines.append(
            "| model | node | conc | ttft p50 | ttft p95 | gen tok/s p50 | "
            "total p50 | thr req/s | thr tok/s | ok/fail |"
        )
        lines.append(
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        for r in rows:
            if not r.available:
                lines.append(
                    f"| {r.model} | {r.node} | {r.concurrency} | - | - | - | - | - | - | "
                    f"_{r.error or 'unavailable'}_ |"
                )
                continue
            lines.append(
                f"| {r.model} | {r.node} | {r.concurrency} | "
                f"{_fmt(r.ttft.p50 if r.ttft else None, 'ms', 0)} | "
                f"{_fmt(r.ttft.p95 if r.ttft else None, 'ms', 0)} | "
                f"{_fmt(r.gen_tok_s.p50 if r.gen_tok_s else None, '', 1)} | "
                f"{_fmt(r.total.p50 if r.total else None, 'ms', 0)} | "
                f"{(_fmt(r.throughput_req_s, '', 2) if r.concurrency > 1 else '-')} | "
                f"{(_fmt(r.throughput_tok_s, '', 1) if r.concurrency > 1 else '-')} | "
                f"{r.successes}/{r.failures} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def write_bundle(
    out_dir: Path,
    results: list[CellRun],
    meta: dict,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = meta.get("timestamp") or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"bench-{ts}.json"
    md_path = out_dir / f"bench-{ts}.md"

    json_path.write_text(json.dumps(serialize_json(results, meta), indent=2))
    md_path.write_text(render_markdown(results, meta))

    # Also overwrite a `latest` pair for ollarouter to consume.
    (out_dir / "bench-latest.json").write_text(json_path.read_text())
    (out_dir / "bench-latest.md").write_text(md_path.read_text())

    return json_path, md_path
