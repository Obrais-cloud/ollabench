"""Config loading — accepts ollafleet-compatible fleet.yml plus a bench.yml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .client import NodeConfig


FLEET_SEARCH = (
    Path("./fleet.yml"),
    Path("./fleet.yaml"),
    Path.home() / ".config" / "ollafleet" / "fleet.yml",
    Path.home() / ".config" / "ollabench" / "fleet.yml",
)


def find_fleet_config(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"fleet config not found: {p}")
        return p
    for p in FLEET_SEARCH:
        if p.exists():
            return p
    raise FileNotFoundError(
        "No fleet.yml found. Pass --fleet PATH or create ./fleet.yml "
        "(ollafleet-compatible schema)."
    )


def load_fleet(path: Path) -> list[NodeConfig]:
    data = yaml.safe_load(path.read_text()) or {}
    nodes_raw = data.get("nodes", [])
    nodes = [
        NodeConfig(
            name=n["name"],
            host=n["host"].rstrip("/"),
            tags=list(n.get("tags", [])),
        )
        for n in nodes_raw
    ]
    if not nodes:
        raise ValueError(f"fleet config {path} has no nodes")
    return nodes


@dataclass
class TargetSpec:
    model: str
    node: str = "*"  # node name, "*", or "tag:<tag>"


@dataclass
class BenchPlan:
    targets: list[TargetSpec]
    scenarios: list[str]
    iterations: int = 5
    warmup: int = 1
    concurrency: list[int] = field(default_factory=lambda: [1])
    timeout_s: float = 600.0
    extra_options: dict = field(default_factory=dict)


def load_bench(path: Path) -> BenchPlan:
    data = yaml.safe_load(path.read_text()) or {}

    targets_raw = data.get("targets") or []
    if not targets_raw:
        raise ValueError(f"{path}: no targets")
    targets: list[TargetSpec] = []
    for t in targets_raw:
        if isinstance(t, str):
            # "model@node" or just "model"
            if "@" in t:
                model, node = t.split("@", 1)
                targets.append(TargetSpec(model=model.strip(), node=node.strip()))
            else:
                targets.append(TargetSpec(model=t.strip()))
        else:
            targets.append(
                TargetSpec(model=t["model"], node=str(t.get("node", "*")))
            )

    scenarios = list(data.get("scenarios") or ["short", "medium", "long_gen"])
    concurrency = list(data.get("concurrency") or [1])

    return BenchPlan(
        targets=targets,
        scenarios=scenarios,
        iterations=int(data.get("iterations", 5)),
        warmup=int(data.get("warmup", 1)),
        concurrency=concurrency,
        timeout_s=float(data.get("timeout_s", 600.0)),
        extra_options=dict(data.get("options") or {}),
    )


def expand_targets(
    targets: list[TargetSpec], nodes: list[NodeConfig]
) -> list[tuple[str, NodeConfig]]:
    pairs: list[tuple[str, NodeConfig]] = []
    by_name = {n.name: n for n in nodes}
    for t in targets:
        if t.node == "*":
            for n in nodes:
                pairs.append((t.model, n))
        elif t.node.startswith("tag:"):
            tag = t.node[4:]
            for n in nodes:
                if tag in n.tags:
                    pairs.append((t.model, n))
        else:
            n = by_name.get(t.node)
            if n is None:
                raise ValueError(
                    f"target {t.model}@{t.node}: node {t.node!r} not in fleet"
                )
            pairs.append((t.model, n))
    # de-dup preserving order
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, NodeConfig]] = []
    for model, node in pairs:
        key = (model, node.name)
        if key in seen:
            continue
        seen.add(key)
        out.append((model, node))
    return out
