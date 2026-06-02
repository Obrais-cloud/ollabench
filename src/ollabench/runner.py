"""Benchmark runner.

For each (model, node, scenario, concurrency) it issues `iterations` requests,
records per-sample timings, then summarizes them into percentile stats.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import anyio

from .client import GenerateSample, NodeConfig, OllamaNode
from .config import BenchPlan
from .scenarios import Scenario
from .stats import Summary, summarize


@dataclass
class CellRun:
    """One (model, node, scenario, concurrency) result block."""

    model: str
    node: str
    scenario: str
    concurrency: int

    available: bool = True
    error: str | None = None

    samples: list[GenerateSample] = field(default_factory=list)
    cold_sample: GenerateSample | None = None

    wall_run_ms: float = 0.0  # wall clock for the entire timed batch
    successes: int = 0
    failures: int = 0

    # Computed in finalize().
    ttft: Summary | None = None
    total: Summary | None = None
    gen_tok_s: Summary | None = None
    wall_tok_s: Summary | None = None
    eval_count: Summary | None = None
    prompt_eval_count: Summary | None = None
    throughput_req_s: float = 0.0       # only meaningful with concurrency > 1
    throughput_tok_s: float = 0.0       # aggregate output tokens/sec

    def finalize(self) -> None:
        good = [s for s in self.samples if s.error is None and s.eval_count > 0]
        self.successes = len(good)
        self.failures = len(self.samples) - self.successes
        if not good:
            return

        ttfts = [s.ttft_ms for s in good if s.ttft_ms is not None]
        self.ttft = summarize(ttfts) if ttfts else None
        self.total = summarize([s.total_ms for s in good])
        self.gen_tok_s = summarize([s.gen_tok_s for s in good])
        self.wall_tok_s = summarize([s.wall_tok_s for s in good])
        self.eval_count = summarize([float(s.eval_count) for s in good])
        self.prompt_eval_count = summarize(
            [float(s.prompt_eval_count) for s in good]
        )

        if self.wall_run_ms > 0:
            self.throughput_req_s = self.successes / (self.wall_run_ms / 1000.0)
            total_out = sum(s.eval_count for s in good)
            self.throughput_tok_s = total_out / (self.wall_run_ms / 1000.0)

    def as_dict(self) -> dict:
        def _s(x: Summary | None) -> dict | None:
            return x.as_dict() if x is not None else None

        return {
            "model": self.model,
            "node": self.node,
            "scenario": self.scenario,
            "concurrency": self.concurrency,
            "available": self.available,
            "error": self.error,
            "samples": len(self.samples),
            "successes": self.successes,
            "failures": self.failures,
            "wall_run_ms": round(self.wall_run_ms, 3),
            "throughput_req_s": round(self.throughput_req_s, 3),
            "throughput_tok_s": round(self.throughput_tok_s, 3),
            "cold_ttft_ms": (
                round(self.cold_sample.ttft_ms, 3)
                if self.cold_sample and self.cold_sample.ttft_ms is not None
                else None
            ),
            "ttft_ms": _s(self.ttft),
            "total_ms": _s(self.total),
            "gen_tok_s": _s(self.gen_tok_s),
            "wall_tok_s": _s(self.wall_tok_s),
            "eval_count": _s(self.eval_count),
            "prompt_eval_count": _s(self.prompt_eval_count),
        }


ProgressCb = Callable[[CellRun], None]


async def _run_one(
    node_client: OllamaNode,
    model: str,
    scenario: Scenario,
    plan: BenchPlan,
) -> GenerateSample:
    return await node_client.generate(
        model,
        scenario.prompt,
        num_predict=scenario.num_predict,
        options=plan.extra_options or None,
    )


async def _run_concurrent(
    node_client: OllamaNode,
    model: str,
    scenario: Scenario,
    plan: BenchPlan,
    concurrency: int,
    iterations: int,
) -> tuple[list[GenerateSample], float]:
    """Issue `iterations` requests with at most `concurrency` in flight."""

    sem = anyio.Semaphore(concurrency)
    samples: list[GenerateSample] = []

    async def task(i: int) -> None:
        async with sem:
            s = await _run_one(node_client, model, scenario, plan)
            samples.append(s)

    start = time.monotonic()
    async with anyio.create_task_group() as tg:
        for i in range(iterations):
            tg.start_soon(task, i)
    wall = (time.monotonic() - start) * 1000
    return samples, wall


async def run_cell(
    node_client: OllamaNode,
    model: str,
    scenario: Scenario,
    plan: BenchPlan,
    concurrency: int,
) -> CellRun:
    cell = CellRun(
        model=model,
        node=node_client.config.name,
        scenario=scenario.name,
        concurrency=concurrency,
    )

    if not await node_client.has_model(model):
        cell.available = False
        cell.error = f"model {model!r} not installed on node"
        return cell

    # Warmup — also captures cold-start TTFT.
    for i in range(plan.warmup):
        cold = await _run_one(node_client, model, scenario, plan)
        if i == 0:
            cell.cold_sample = cold
        if cold.error:
            cell.error = f"warmup error: {cold.error}"
            return cell

    if concurrency <= 1:
        # Sequential timed batch — gives clean per-sample numbers.
        start = time.monotonic()
        for _ in range(plan.iterations):
            s = await _run_one(node_client, model, scenario, plan)
            cell.samples.append(s)
        cell.wall_run_ms = (time.monotonic() - start) * 1000
    else:
        samples, wall = await _run_concurrent(
            node_client, model, scenario, plan, concurrency, plan.iterations
        )
        cell.samples = samples
        cell.wall_run_ms = wall

    cell.finalize()
    return cell


async def run_plan(
    pairs: list[tuple[str, NodeConfig]],
    scenarios: list[Scenario],
    plan: BenchPlan,
    progress: ProgressCb | None = None,
) -> list[CellRun]:
    """Run the full plan. Per node, cells run sequentially to avoid skewing
    measurements with cross-cell interference. Different nodes run in parallel."""

    results: list[CellRun] = []

    by_node: dict[str, tuple[NodeConfig, list[str]]] = {}
    for model, node in pairs:
        slot = by_node.setdefault(node.name, (node, []))
        slot[1].append(model)

    async def per_node(node: NodeConfig, models: list[str]) -> None:
        client = OllamaNode(node, timeout=plan.timeout_s)
        try:
            ok, _ = await client.ping()
            if not ok:
                for model in models:
                    for sc in scenarios:
                        for c in plan.concurrency:
                            cell = CellRun(
                                model=model,
                                node=node.name,
                                scenario=sc.name,
                                concurrency=c,
                                available=False,
                                error="node unreachable",
                            )
                            results.append(cell)
                            if progress:
                                progress(cell)
                return

            for model in models:
                for sc in scenarios:
                    for c in plan.concurrency:
                        cell = await run_cell(client, model, sc, plan, c)
                        results.append(cell)
                        if progress:
                            progress(cell)
        finally:
            await client.close()

    async with anyio.create_task_group() as tg:
        for node, models in by_node.values():
            tg.start_soon(per_node, node, models)

    return results
