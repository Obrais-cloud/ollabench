"""Core benchmarking engine for Ollama inference."""

import json
import time
import statistics
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict


# Standard benchmark prompts by category
PROMPTS = {
    "short": "What is 2 + 2?",
    "medium": "Explain the difference between TCP and UDP in networking. Be concise.",
    "long": (
        "Write a detailed technical explanation of how garbage collection works "
        "in modern programming languages. Cover reference counting, mark-and-sweep, "
        "generational collection, and compare tradeoffs. Include examples."
    ),
    "code": "Write a Python function that implements binary search on a sorted list. Include docstring and type hints.",
    "reasoning": (
        "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer "
        "have left? Explain your reasoning step by step."
    ),
}


@dataclass
class BenchResult:
    """Result of a single benchmark run."""
    host: str
    model: str
    prompt_category: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    time_to_first_token: float  # seconds
    total_time: float  # seconds
    tokens_per_second: float
    prompt_eval_rate: float  # tokens/sec for prompt processing
    success: bool
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelBenchmark:
    """Aggregated benchmark results for a model on a host."""
    host: str
    model: str
    results: list[BenchResult] = field(default_factory=list)

    @property
    def successful(self) -> list[BenchResult]:
        return [r for r in self.results if r.success]

    @property
    def avg_tokens_per_second(self) -> float:
        s = self.successful
        return statistics.mean(r.tokens_per_second for r in s) if s else 0.0

    @property
    def median_tokens_per_second(self) -> float:
        s = self.successful
        return statistics.median(r.tokens_per_second for r in s) if s else 0.0

    @property
    def p95_latency(self) -> float:
        s = self.successful
        if not s:
            return 0.0
        times = sorted(r.total_time for r in s)
        idx = int(len(times) * 0.95)
        return times[min(idx, len(times) - 1)]

    @property
    def avg_ttft(self) -> float:
        s = self.successful
        return statistics.mean(r.time_to_first_token for r in s) if s else 0.0

    @property
    def avg_prompt_eval_rate(self) -> float:
        s = self.successful
        rates = [r.prompt_eval_rate for r in s if r.prompt_eval_rate > 0]
        return statistics.mean(rates) if rates else 0.0

    def summary(self) -> dict:
        return {
            "host": self.host,
            "model": self.model,
            "runs": len(self.results),
            "successes": len(self.successful),
            "avg_tok_s": round(self.avg_tokens_per_second, 1),
            "median_tok_s": round(self.median_tokens_per_second, 1),
            "p95_latency_s": round(self.p95_latency, 2),
            "avg_ttft_s": round(self.avg_ttft, 3),
            "avg_prompt_eval_tok_s": round(self.avg_prompt_eval_rate, 1),
        }


def discover_models(host: str, timeout: float = 5.0) -> list[str]:
    """List models available on an Ollama host."""
    url = f"{host.rstrip('/')}/api/tags"
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def check_host(host: str, timeout: float = 3.0) -> bool:
    """Check if an Ollama host is reachable."""
    try:
        req = urllib.request.Request(f"{host.rstrip('/')}/api/tags")
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except Exception:
        return False


def run_single_bench(
    host: str,
    model: str,
    prompt: str,
    prompt_category: str = "custom",
    timeout: float = 120.0,
) -> BenchResult:
    """Run a single benchmark against an Ollama host."""
    url = f"{host.rstrip('/')}/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"num_predict": 512},
    }).encode()

    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})

    start = time.perf_counter()
    ttft = 0.0
    total_eval_tokens = 0
    total_prompt_tokens = 0
    prompt_eval_duration = 0
    eval_duration = 0
    first_token = True

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for line in resp:
                if not line.strip():
                    continue
                chunk = json.loads(line)

                if first_token and chunk.get("response", ""):
                    ttft = time.perf_counter() - start
                    first_token = False

                if chunk.get("done"):
                    total_eval_tokens = chunk.get("eval_count", 0)
                    total_prompt_tokens = chunk.get("prompt_eval_count", 0)
                    prompt_eval_duration = chunk.get("prompt_eval_duration", 0)
                    eval_duration = chunk.get("eval_duration", 0)
                    break

        total_time = time.perf_counter() - start

        # Ollama reports durations in nanoseconds
        eval_sec = eval_duration / 1e9 if eval_duration else total_time
        prompt_eval_sec = prompt_eval_duration / 1e9 if prompt_eval_duration else 0

        tok_s = total_eval_tokens / eval_sec if eval_sec > 0 else 0
        prompt_rate = total_prompt_tokens / prompt_eval_sec if prompt_eval_sec > 0 else 0

        return BenchResult(
            host=host,
            model=model,
            prompt_category=prompt_category,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_eval_tokens,
            total_tokens=total_prompt_tokens + total_eval_tokens,
            time_to_first_token=round(ttft, 4),
            total_time=round(total_time, 4),
            tokens_per_second=round(tok_s, 2),
            prompt_eval_rate=round(prompt_rate, 2),
            success=True,
        )

    except Exception as e:
        total_time = time.perf_counter() - start
        return BenchResult(
            host=host,
            model=model,
            prompt_category=prompt_category,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            time_to_first_token=0,
            total_time=round(total_time, 4),
            tokens_per_second=0,
            prompt_eval_rate=0,
            success=False,
            error=str(e),
        )


def bench_model(
    host: str,
    model: str,
    categories: list[str] | None = None,
    runs: int = 1,
    timeout: float = 120.0,
    on_result=None,
) -> ModelBenchmark:
    """Benchmark a model across prompt categories."""
    cats = categories or ["short", "medium", "code"]
    benchmark = ModelBenchmark(host=host, model=model)

    for cat in cats:
        prompt = PROMPTS.get(cat, cat)  # use as literal prompt if not a known category
        for run_idx in range(runs):
            result = run_single_bench(host, model, prompt, cat, timeout)
            benchmark.results.append(result)
            if on_result:
                on_result(result, run_idx + 1, runs)

    return benchmark
