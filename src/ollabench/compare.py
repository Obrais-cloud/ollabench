"""Compare two ollabench result files: A (baseline) vs B (new).

Used both for upgrade testing (new model, new node, new Ollama version) and
regression checks (did anything get >X% slower since last run)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Delta:
    key: str          # model|node|scenario
    metric: str       # e.g. ttft_p50_ms
    a: float
    b: float
    pct: float        # (b - a) / a * 100; positive = b is larger

    @property
    def regressed(self) -> bool:
        # For latency metrics, larger = worse. For tok/s, larger = better.
        if self.metric.endswith("ms"):
            return self.pct > 0
        return self.pct < 0

    @property
    def improved(self) -> bool:
        return (not self.regressed) and abs(self.pct) > 0.01


def _load_matrix(path: Path) -> dict:
    data = json.loads(path.read_text())
    return data.get("matrix") or {}


def diff(a_path: Path, b_path: Path, metrics: list[str] | None = None) -> list[Delta]:
    a = _load_matrix(a_path)
    b = _load_matrix(b_path)
    metrics = metrics or [
        "ttft_p50_ms",
        "ttft_p95_ms",
        "gen_tok_s_p50",
        "total_p50_ms",
    ]

    out: list[Delta] = []
    for key, av in a.items():
        bv = b.get(key)
        if bv is None:
            continue
        for m in metrics:
            x = av.get(m)
            y = bv.get(m)
            if x is None or y is None or x == 0:
                continue
            pct = (y - x) / x * 100.0
            out.append(Delta(key=key, metric=m, a=x, b=y, pct=pct))
    return out


def filter_significant(deltas: list[Delta], threshold_pct: float = 10.0) -> list[Delta]:
    return [d for d in deltas if abs(d.pct) >= threshold_pct]
