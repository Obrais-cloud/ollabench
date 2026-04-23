# ollabench

Fleet-wide inference benchmarking for Ollama. Find which node runs which model fastest.

When you're running a heterogeneous fleet (Mac Studio M2 Ultra, Corsair i300, Alienware 4090...), you need hard numbers — not guesses — for model placement decisions. ollabench gives you tokens/sec, time-to-first-token, latency percentiles, and prompt eval rates across every model on every node.

Feeds directly into [ollaroute](https://github.com/Obrais-cloud/ollaroute) routing and [ollapark](https://github.com/Obrais-cloud/ollapark) placement decisions.

## Install

```bash
pip install -e .
```

## Quick Start

```bash
# Benchmark a single model
ollabench quick llama3.1:8b

# Benchmark your entire fleet
export OLLAMA_HOSTS="http://mac-studio:11434,http://corsair:11434,http://alienware:11434"
ollabench run

# Benchmark specific models across fleet
ollabench run -m "llama3.1:8b,qwen2.5-coder:7b,deepseek-r1:14b"

# Multiple runs for statistical confidence
ollabench run -m "llama3.1:8b" -r 3

# All prompt categories
ollabench run -m "llama3.1:8b" -c "short,medium,long,code,reasoning"

# Export to JSON
ollabench run -m "llama3.1:8b" --json results.json

# Detailed per-run breakdown
ollabench run -m "llama3.1:8b" --detail
```

## Commands

### `ollabench run`

Full fleet benchmark. Discovers all nodes, benchmarks models, generates comparison reports.

```
Options:
  --hosts          Comma-separated hosts (or set OLLAMA_HOSTS env var)
  -m, --models     Comma-separated models (default: all on each node)
  -c, --categories Prompt categories: short,medium,long,code,reasoning
  -r, --runs       Runs per prompt (default: 1, use 3+ for reliable stats)
  -t, --timeout    Timeout per request in seconds (default: 120)
  -d, --detail     Show per-run breakdown
  -j, --json       Export results to JSON file
```

### `ollabench quick <model>`

Fast single-model benchmark on one host. Good for quick checks.

### `ollabench status`

Show fleet status — which nodes are online, what models are available, and which models exist on multiple nodes.

### `ollabench prompts`

List the built-in benchmark prompt categories.

## What It Measures

| Metric | Description |
|--------|-------------|
| **tok/s** | Generation throughput (eval tokens / eval time) |
| **TTFT** | Time to first token — how fast the model starts responding |
| **P95 latency** | 95th percentile total request time |
| **Prompt eval tok/s** | Prompt processing speed (context ingestion) |

## Output

### Summary Table

```
Host             | Model          | Runs | Avg tok/s | Med tok/s | P95 lat(s) | TTFT(s) | Prompt eval tok/s
-----------------+----------------+------+-----------+-----------+------------+---------+------------------
mac-studio:11434 | llama3.1:8b    | 3    | 42.5      | 43.1      | 12.34      | 0.891   | 312.4
corsair:11434    | llama3.1:8b    | 3    | 38.2      | 37.9      | 14.21      | 1.023   | 289.1
alienware:11434  | llama3.1:8b    | 3    | 67.8      | 68.2      | 8.45       | 0.234   | 523.7
```

### Fleet Leaderboard

Shows optimal model-to-node placement based on throughput:

```
Model           | Best Host         | tok/s | TTFT(s) | Runner-up        | Runner-up tok/s
----------------+-------------------+-------+---------+------------------+----------------
llama3.1:8b     | alienware:11434   | 67.8  | 0.234   | mac-studio:11434 | 42.5
qwen2.5-coder:7b| mac-studio:11434  | 51.2  | 0.567   | corsair:11434    | 44.3
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OLLAMA_HOSTS` | Comma-separated list of Ollama endpoints |
| `OLLAMA_HOST` | Single Ollama endpoint (fallback, default: `http://localhost:11434`) |

## Zero Dependencies

Pure Python stdlib. No pip dependencies. Works anywhere Python 3.10+ runs.

## Part of the Ollama Fleet Toolkit

ollabench is one of 20+ tools for managing local AI infrastructure:

- **ollaroute** — smart model routing (use ollabench data to configure)
- **ollapark** — model placement optimization (use ollabench data to decide)
- **olladash** — fleet web dashboard
- **ollamatop** — TUI monitoring
- **ollameter** — usage tracking proxy
- **ollarena** — output quality comparison

## License

MIT
