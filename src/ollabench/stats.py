"""Percentile + aggregation helpers used by runner and report."""

from __future__ import annotations

from dataclasses import dataclass


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile. `p` is 0-100. Returns 0.0 for empty."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass
class Summary:
    n: int
    mean: float
    p50: float
    p95: float
    p99: float
    min: float
    max: float

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "mean": round(self.mean, 3),
            "p50": round(self.p50, 3),
            "p95": round(self.p95, 3),
            "p99": round(self.p99, 3),
            "min": round(self.min, 3),
            "max": round(self.max, 3),
        }


def summarize(values: list[float]) -> Summary:
    if not values:
        return Summary(n=0, mean=0.0, p50=0.0, p95=0.0, p99=0.0, min=0.0, max=0.0)
    return Summary(
        n=len(values),
        mean=mean(values),
        p50=percentile(values, 50),
        p95=percentile(values, 95),
        p99=percentile(values, 99),
        min=min(values),
        max=max(values),
    )
