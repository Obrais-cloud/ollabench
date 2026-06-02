# ollabench

**Performance benchmark matrix for an Ollama fleet.**

Where [ollacaps] tells you what each `(model, node)` cell *can do*, ollabench
tells you *how fast it does it*. Output is a JSON matrix that
[ollarouter] can consume to make latency- and throughput-aware routing
decisions, plus a markdown report for humans.

Measures, per `(model, node, scenario, concurrency)` cell:

- **TTFT** — wall-clock time-to-first-token (P50, P95, P99)
- **gen tok/s** — sustained generation rate from Ollama's `eval_duration`
- **wall tok/s** — generation rate from the client's POV (TTFT excluded)
- **total latency** — request lifetime (P50, P95, P99)
- **throughput** — agg req/s and tok/s under concurrent load
- **cold TTFT** — first-request TTFT (model load cost)
- **failure rate** — per cell

[ollacaps]: https://github.com/braisrevalderia/ollacaps
[ollarouter]: https://github.com/braisrevalderia/ollarouter

## Install

```bash
pip install -e .
```

## Configure

Reuses ollafleet's `fleet.yml` schema, so if you already have one, just point
at it (or drop one in `./fleet.yml`):

```yaml
nodes:
  - name: mac
    host: http://localhost:11434
    tags: [local, metal]
  - name: rig
    host: http://10.0.0.50:11434
    tags: [gpu, cuda]
```

A bench plan picks targets and scenarios:

```yaml
# bench.yml
targets:
  - llama3.2:3b              # runs on every node
  - qwen2.5:7b@rig           # specific node
  - llama3.1:8b@tag:gpu      # all nodes with the `gpu` tag

scenarios: [tiny, short, medium, long_gen]

iterations: 5
warmup: 1
concurrency: [1, 4]

options:
  temperature: 0
  seed: 7
  num_ctx: 4096
```

## Run

```bash
# Full plan from a file
ollabench run --fleet fleet.yml --bench bench.yml --out ./bench-out

# Quick one-shot from flags
ollabench run --model llama3.2:3b --scenario short -n 5 -c 1 -c 4

# List built-in scenarios
ollabench scenarios

# Re-render a result file
ollabench report ./bench-out/bench-latest.json --format markdown

# Regression check vs a saved baseline
ollabench compare ./baselines/2026-05.json ./bench-out/bench-latest.json --threshold 10
```

## Scenarios

| name          | num_predict | shape                                          |
|---------------|-------------|------------------------------------------------|
| `tiny`        | 8           | smallest possible — TTFT + warm cost           |
| `short`       | 80          | short prompt + short gen (chat-style)          |
| `medium`      | 200         | ~500-token prompt + medium gen (summarization) |
| `long_prompt` | 120         | ~2k-token prompt + short gen (prompt-eval)     |
| `long_gen`    | 1200        | short prompt + long gen (sustained tok/s)      |

Prompts are deterministic so prompt-eval is identical across runs.

## Output

Each run writes a timestamped pair plus a stable `bench-latest.{json,md}` for
downstream tools:

```
bench-out/
├── bench-20260602T060633Z.json
├── bench-20260602T060633Z.md
├── bench-latest.json    # for ollarouter
└── bench-latest.md
```

The JSON has three top-level keys:

- `meta` — run config + timestamp
- `results` — full per-cell stats (every summary in mean/p50/p95/p99/min/max form)
- `matrix` — compact `model|node|scenario → headline metrics`, the shape
  ollarouter eats:

```json
"qwen2.5:7b|rig|short": {
  "ttft_p50_ms": 145.2,
  "ttft_p95_ms": 178.0,
  "gen_tok_s_p50": 92.3,
  "gen_tok_s_p95": 95.1,
  "total_p50_ms": 880.0,
  "cold_ttft_ms": 12300.0
}
```

## How it plays with the rest of the stack

- **ollarouter** — read `bench-latest.json` and pick `(model, node)` that
  minimizes `ttft_p50_ms + expected_tokens / gen_tok_s_p50` for a given task
  shape. `cold_ttft_ms` flags cells that need `ollawarm` to stay warm.
- **ollawarm** — prioritize warming on cells with high `cold_ttft_ms` and
  high `ttft_p50_ms`.
- **ollaregress** — snapshot bench results as a perf baseline; CI compares
  against it via `ollabench compare`.
- **ollacost** — combine `gen_tok_s_p50` with node wattage to derive
  $/Mtoken per cell.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

## Schema (router contract)

The `matrix` block is the stable contract. Schema:

```
matrix: {
  "<model>|<node>|<scenario>": {
    ttft_p50_ms:   number  # client wall-clock
    ttft_p95_ms:   number
    gen_tok_s_p50: number  # from Ollama eval_duration
    gen_tok_s_p95: number
    total_p50_ms:  number
    cold_ttft_ms:  number | null  # first-request TTFT, null if no warmup
  }
}
```

Adding new metrics is additive. Removing keys is a breaking change — bump
`meta.ollabench_version`.

## License

MIT.
