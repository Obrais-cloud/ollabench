"""CLI interface for ollabench."""

import argparse
import json
import sys
import time

from ollabench import __version__
from ollabench.bench import PROMPTS, bench_model
from ollabench.fleet import bench_fleet, discover_fleet, get_fleet_hosts
from ollabench.report import (
    export_json,
    print_comparison,
    print_detail,
    print_leaderboard,
    print_summary,
)


def _progress_callback(result, run_idx, total_runs):
    """Print progress for each benchmark run."""
    status = "OK" if result.success else f"FAIL ({result.error})"
    tok_s = f"{result.tokens_per_second} tok/s" if result.success else "-"
    host = result.host.replace("http://", "").replace("https://", "")
    print(f"  [{run_idx}/{total_runs}] {result.prompt_category:8s} → {tok_s:>12s}  ttft={result.time_to_first_token}s  {status}")


def _model_start_callback(host, model):
    """Print when starting a new model benchmark."""
    label = host.replace("http://", "").replace("https://", "")
    print(f"\n▶ {label} / {model}")


def cmd_run(args):
    """Run benchmarks."""
    hosts = args.hosts.split(",") if args.hosts else None
    models = args.models.split(",") if args.models else None
    categories = args.categories.split(",") if args.categories else None

    print(f"ollabench v{__version__}")
    print(f"Runs per prompt: {args.runs}")
    print(f"Categories: {categories or ['short', 'medium', 'code']}")
    print(f"Timeout: {args.timeout}s")
    print()

    # Discover fleet
    fleet_hosts = hosts or get_fleet_hosts()
    nodes = discover_fleet(fleet_hosts)
    online = [n for n in nodes if n.online]

    if not online:
        print("No Ollama nodes reachable. Set OLLAMA_HOSTS or OLLAMA_HOST.", file=sys.stderr)
        sys.exit(1)

    print(f"Fleet: {len(online)}/{len(nodes)} nodes online")
    for n in nodes:
        status = f"v{n.version} — {len(n.models)} models" if n.online else "OFFLINE"
        print(f"  {'●' if n.online else '○'} {n.label:30s} {status}")
    print()

    start = time.perf_counter()

    benchmarks = bench_fleet(
        hosts=fleet_hosts,
        models=models,
        categories=categories,
        runs=args.runs,
        timeout=args.timeout,
        on_result=_progress_callback,
        on_model_start=_model_start_callback,
    )

    elapsed = time.perf_counter() - start
    print(f"\nCompleted in {elapsed:.1f}s\n")

    if not benchmarks:
        print("No benchmarks completed. Check that models are available on your nodes.")
        return

    # Summary table
    print_summary(benchmarks)

    # Comparison (only useful with multiple hosts or models)
    if len(benchmarks) > 1:
        print_comparison(benchmarks)
        print_leaderboard(benchmarks)

    # Detail if requested
    if args.detail:
        print_detail(benchmarks)

    # JSON export
    if args.json:
        with open(args.json, "w") as f:
            f.write(export_json(benchmarks))
        print(f"\nResults exported to {args.json}")


def cmd_quick(args):
    """Quick single-model benchmark."""
    host = args.host or get_fleet_hosts()[0]
    model = args.model

    print(f"ollabench quick — {model} @ {host}")
    print()

    benchmarks = bench_model(
        host=host,
        model=model,
        categories=["short", "medium", "code"],
        runs=1,
        timeout=args.timeout,
        on_result=_progress_callback,
    )

    print()
    print_summary([benchmarks])

    if args.detail:
        print_detail([benchmarks])


def cmd_status(args):
    """Show fleet status and available models."""
    hosts = args.hosts.split(",") if args.hosts else None
    fleet_hosts = hosts or get_fleet_hosts()
    nodes = discover_fleet(fleet_hosts)

    print(f"ollabench v{__version__} — Fleet Status\n")

    online = [n for n in nodes if n.online]
    print(f"Nodes: {len(online)}/{len(nodes)} online\n")

    for n in nodes:
        status = "ONLINE" if n.online else "OFFLINE"
        print(f"{'●' if n.online else '○'} {n.label:30s} {status}")
        if n.online:
            print(f"  Version: {n.version}")
            print(f"  Models ({len(n.models)}):")
            for m in sorted(n.models):
                print(f"    - {m}")
        print()

    # Show models available across nodes
    if len(online) > 1:
        all_models: dict[str, list[str]] = {}
        for n in online:
            for m in n.models:
                all_models.setdefault(m, []).append(n.label)

        shared = {m: hosts for m, hosts in all_models.items() if len(hosts) > 1}
        if shared:
            print("Models available on multiple nodes:")
            for m, hosts in sorted(shared.items()):
                print(f"  {m}: {', '.join(hosts)}")
            print()


def cmd_prompts(args):
    """List available benchmark prompt categories."""
    print("Built-in benchmark prompts:\n")
    for cat, prompt in PROMPTS.items():
        preview = prompt[:80] + "..." if len(prompt) > 80 else prompt
        print(f"  {cat:12s}  {preview}")
    print()


def main():
    parser = argparse.ArgumentParser(
        prog="ollabench",
        description="Fleet-wide inference benchmarking for Ollama",
    )
    parser.add_argument("--version", action="version", version=f"ollabench {__version__}")

    sub = parser.add_subparsers(dest="command")

    # run
    p_run = sub.add_parser("run", help="Run fleet-wide benchmarks")
    p_run.add_argument("--hosts", help="Comma-separated Ollama hosts (or set OLLAMA_HOSTS)")
    p_run.add_argument("--models", "-m", help="Comma-separated models to benchmark (default: all on each node)")
    p_run.add_argument("--categories", "-c", help="Comma-separated prompt categories (default: short,medium,code)")
    p_run.add_argument("--runs", "-r", type=int, default=1, help="Runs per prompt category (default: 1)")
    p_run.add_argument("--timeout", "-t", type=float, default=120.0, help="Timeout per request in seconds (default: 120)")
    p_run.add_argument("--detail", "-d", action="store_true", help="Show detailed per-run results")
    p_run.add_argument("--json", "-j", help="Export results to JSON file")
    p_run.set_defaults(func=cmd_run)

    # quick
    p_quick = sub.add_parser("quick", help="Quick single-model benchmark")
    p_quick.add_argument("model", help="Model name to benchmark")
    p_quick.add_argument("--host", help="Ollama host (default: OLLAMA_HOST or localhost:11434)")
    p_quick.add_argument("--timeout", "-t", type=float, default=120.0, help="Timeout per request (default: 120)")
    p_quick.add_argument("--detail", "-d", action="store_true", help="Show detailed results")
    p_quick.set_defaults(func=cmd_quick)

    # status
    p_status = sub.add_parser("status", help="Show fleet status and models")
    p_status.add_argument("--hosts", help="Comma-separated Ollama hosts (or set OLLAMA_HOSTS)")
    p_status.set_defaults(func=cmd_status)

    # prompts
    p_prompts = sub.add_parser("prompts", help="List benchmark prompt categories")
    p_prompts.set_defaults(func=cmd_prompts)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
