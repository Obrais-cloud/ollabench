"""Fleet discovery and parallel benchmarking across multiple Ollama nodes."""

import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from ollabench.bench import (
    BenchResult,
    ModelBenchmark,
    bench_model,
    check_host,
    discover_models,
)


DEFAULT_HOST = "http://localhost:11434"


@dataclass
class NodeInfo:
    """Information about a discovered Ollama node."""
    host: str
    online: bool
    models: list[str]
    version: str = ""

    @property
    def label(self) -> str:
        """Short label for display."""
        return self.host.replace("http://", "").replace("https://", "")


def get_fleet_hosts() -> list[str]:
    """Get list of fleet hosts from env or config.

    Reads OLLAMA_HOSTS (comma-separated) or falls back to OLLAMA_HOST or localhost.
    """
    hosts_env = os.environ.get("OLLAMA_HOSTS", "")
    if hosts_env:
        return [h.strip() for h in hosts_env.split(",") if h.strip()]

    single = os.environ.get("OLLAMA_HOST", DEFAULT_HOST)
    return [single]


def get_node_info(host: str) -> NodeInfo:
    """Get info about a single node."""
    if not check_host(host):
        return NodeInfo(host=host, online=False, models=[])

    models = discover_models(host)

    # Try to get version
    version = ""
    try:
        req = urllib.request.Request(f"{host.rstrip('/')}/api/version")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            version = data.get("version", "")
    except Exception:
        pass

    return NodeInfo(host=host, online=True, models=models, version=version)


def discover_fleet(hosts: list[str] | None = None) -> list[NodeInfo]:
    """Discover all nodes in the fleet in parallel."""
    hosts = hosts or get_fleet_hosts()

    with ThreadPoolExecutor(max_workers=min(len(hosts), 10)) as pool:
        futures = {pool.submit(get_node_info, h): h for h in hosts}
        nodes = []
        for future in as_completed(futures):
            nodes.append(future.result())

    # Sort: online first, then by host
    nodes.sort(key=lambda n: (not n.online, n.host))
    return nodes


def bench_fleet(
    hosts: list[str] | None = None,
    models: list[str] | None = None,
    categories: list[str] | None = None,
    runs: int = 1,
    timeout: float = 120.0,
    on_result=None,
    on_model_start=None,
) -> list[ModelBenchmark]:
    """Benchmark models across all fleet nodes in parallel.

    If models is None, benchmarks all models found on each node.
    """
    nodes = discover_fleet(hosts)
    online_nodes = [n for n in nodes if n.online]

    if not online_nodes:
        return []

    # Build work items: (host, model) pairs
    work = []
    for node in online_nodes:
        node_models = models if models else node.models
        # Only bench models that exist on this node
        available = set(node.models)
        for m in node_models:
            if m in available:
                work.append((node.host, m))

    results: list[ModelBenchmark] = []

    # Run benchmarks — sequential per model to avoid overloading nodes,
    # but could parallelize across nodes in the future
    for host, model in work:
        if on_model_start:
            on_model_start(host, model)
        bm = bench_model(host, model, categories, runs, timeout, on_result)
        results.append(bm)

    return results
