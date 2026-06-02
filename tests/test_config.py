from pathlib import Path

import pytest

from ollabench.client import NodeConfig
from ollabench.config import (
    BenchPlan,
    TargetSpec,
    expand_targets,
    load_bench,
    load_fleet,
)


def test_load_fleet(tmp_path: Path):
    p = tmp_path / "fleet.yml"
    p.write_text(
        "nodes:\n"
        "  - name: a\n"
        "    host: http://a:11434\n"
        "  - name: b\n"
        "    host: http://b:11434/\n"
        "    tags: [gpu]\n"
    )
    nodes = load_fleet(p)
    assert [n.name for n in nodes] == ["a", "b"]
    assert nodes[1].host == "http://b:11434"
    assert nodes[1].tags == ["gpu"]


def test_load_fleet_rejects_empty(tmp_path: Path):
    p = tmp_path / "fleet.yml"
    p.write_text("nodes: []\n")
    with pytest.raises(ValueError):
        load_fleet(p)


def test_load_bench_string_targets(tmp_path: Path):
    p = tmp_path / "bench.yml"
    p.write_text(
        "targets:\n"
        "  - llama3.2:3b\n"
        "  - qwen2.5:7b@rig\n"
        "scenarios: [short, medium]\n"
        "iterations: 3\n"
        "concurrency: [1, 4]\n"
    )
    plan = load_bench(p)
    assert plan.iterations == 3
    assert plan.scenarios == ["short", "medium"]
    assert plan.targets[0] == TargetSpec(model="llama3.2:3b", node="*")
    assert plan.targets[1] == TargetSpec(model="qwen2.5:7b", node="rig")
    assert plan.concurrency == [1, 4]


def test_expand_targets_star_and_tag():
    nodes = [
        NodeConfig(name="a", host="http://a"),
        NodeConfig(name="b", host="http://b", tags=["gpu"]),
        NodeConfig(name="c", host="http://c", tags=["gpu"]),
    ]
    targets = [
        TargetSpec(model="m1", node="*"),
        TargetSpec(model="m2", node="tag:gpu"),
        TargetSpec(model="m1", node="a"),  # de-duped
    ]
    pairs = expand_targets(targets, nodes)
    names = [(m, n.name) for m, n in pairs]
    assert ("m1", "a") in names
    assert ("m1", "b") in names
    assert ("m2", "b") in names
    assert ("m2", "c") in names
    assert ("m2", "a") not in names
    assert len([k for k in names if k == ("m1", "a")]) == 1


def test_expand_targets_unknown_node():
    nodes = [NodeConfig(name="a", host="http://a")]
    with pytest.raises(ValueError):
        expand_targets([TargetSpec(model="m1", node="ghost")], nodes)
