"""Report generation — tables, comparisons, and JSON export."""

import json
import sys
from ollabench.bench import ModelBenchmark, BenchResult


def _col_width(rows: list[list[str]], col: int, header: str) -> int:
    """Calculate column width from data + header."""
    max_data = max((len(r[col]) for r in rows), default=0)
    return max(max_data, len(header))


def print_table(headers: list[str], rows: list[list[str]], file=None):
    """Print an aligned ASCII table."""
    f = file or sys.stdout
    widths = [_col_width(rows, i, h) for i, h in enumerate(headers)]

    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep_line = "-+-".join("-" * w for w in widths)

    print(header_line, file=f)
    print(sep_line, file=f)
    for row in rows:
        line = " | ".join(val.ljust(w) for val, w in zip(row, widths))
        print(line, file=f)


def format_host(host: str) -> str:
    """Short host label."""
    return host.replace("http://", "").replace("https://", "")


def print_summary(benchmarks: list[ModelBenchmark], file=None):
    """Print a summary table of all benchmark results."""
    if not benchmarks:
        print("No benchmark results.", file=file)
        return

    headers = ["Host", "Model", "Runs", "Avg tok/s", "Med tok/s", "P95 lat(s)", "TTFT(s)", "Prompt eval tok/s"]
    rows = []

    for bm in benchmarks:
        s = bm.summary()
        rows.append([
            format_host(s["host"]),
            s["model"],
            str(s["runs"]),
            str(s["avg_tok_s"]),
            str(s["median_tok_s"]),
            str(s["p95_latency_s"]),
            str(s["avg_ttft_s"]),
            str(s["avg_prompt_eval_tok_s"]),
        ])

    print_table(headers, rows, file)


def print_comparison(benchmarks: list[ModelBenchmark], file=None):
    """Print a comparison showing which host is fastest for each model."""
    f = file or sys.stdout

    # Group by model
    by_model: dict[str, list[ModelBenchmark]] = {}
    for bm in benchmarks:
        by_model.setdefault(bm.model, []).append(bm)

    for model, bms in sorted(by_model.items()):
        print(f"\n{'='*60}", file=f)
        print(f"  {model}", file=f)
        print(f"{'='*60}", file=f)

        # Sort by avg tokens/sec descending
        ranked = sorted(bms, key=lambda b: b.avg_tokens_per_second, reverse=True)

        headers = ["Rank", "Host", "Avg tok/s", "Med tok/s", "TTFT(s)", "P95 lat(s)"]
        rows = []
        for i, bm in enumerate(ranked):
            s = bm.summary()
            medal = ">>>" if i == 0 else "   "
            rows.append([
                f"{medal} #{i+1}",
                format_host(s["host"]),
                str(s["avg_tok_s"]),
                str(s["median_tok_s"]),
                str(s["avg_ttft_s"]),
                str(s["p95_latency_s"]),
            ])

        print_table(headers, rows, f)

        if len(ranked) >= 2:
            best = ranked[0]
            second = ranked[1]
            if second.avg_tokens_per_second > 0:
                speedup = best.avg_tokens_per_second / second.avg_tokens_per_second
                print(f"\n  Winner: {format_host(best.host)} ({speedup:.1f}x faster)", file=f)
            else:
                print(f"\n  Winner: {format_host(best.host)}", file=f)

    print(file=f)


def print_detail(benchmarks: list[ModelBenchmark], file=None):
    """Print detailed per-run results."""
    f = file or sys.stdout

    for bm in benchmarks:
        print(f"\n--- {format_host(bm.host)} / {bm.model} ---", file=f)
        headers = ["Category", "Prompt tok", "Completion tok", "tok/s", "TTFT(s)", "Total(s)", "Status"]
        rows = []
        for r in bm.results:
            rows.append([
                r.prompt_category,
                str(r.prompt_tokens),
                str(r.completion_tokens),
                str(r.tokens_per_second),
                str(r.time_to_first_token),
                str(r.total_time),
                "OK" if r.success else f"FAIL: {r.error}",
            ])
        print_table(headers, rows, f)


def export_json(benchmarks: list[ModelBenchmark]) -> str:
    """Export all results as JSON."""
    data = {
        "benchmarks": [
            {
                "host": bm.host,
                "model": bm.model,
                "summary": bm.summary(),
                "results": [r.to_dict() for r in bm.results],
            }
            for bm in benchmarks
        ]
    }
    return json.dumps(data, indent=2)


def print_leaderboard(benchmarks: list[ModelBenchmark], file=None):
    """Print a fleet-wide leaderboard: best host for each model."""
    f = file or sys.stdout

    by_model: dict[str, list[ModelBenchmark]] = {}
    for bm in benchmarks:
        by_model.setdefault(bm.model, []).append(bm)

    print("\n  FLEET LEADERBOARD", file=f)
    print("  Optimal model-to-node placement based on throughput\n", file=f)

    headers = ["Model", "Best Host", "tok/s", "TTFT(s)", "Runner-up", "Runner-up tok/s"]
    rows = []

    for model in sorted(by_model):
        bms = sorted(by_model[model], key=lambda b: b.avg_tokens_per_second, reverse=True)
        best = bms[0]
        bs = best.summary()

        if len(bms) >= 2:
            runner = bms[1]
            rs = runner.summary()
            rows.append([
                model,
                format_host(bs["host"]),
                str(bs["avg_tok_s"]),
                str(bs["avg_ttft_s"]),
                format_host(rs["host"]),
                str(rs["avg_tok_s"]),
            ])
        else:
            rows.append([
                model,
                format_host(bs["host"]),
                str(bs["avg_tok_s"]),
                str(bs["avg_ttft_s"]),
                "-",
                "-",
            ])

    print_table(headers, rows, f)
    print(file=f)
