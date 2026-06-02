"""Built-in benchmark scenarios.

Each scenario fixes a prompt shape and an output budget so that results are
comparable across (model, node) cells. Prompts are deterministic — same string
every time — so the prompt-eval portion of the work is identical.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Scenario:
    name: str
    prompt: str
    num_predict: int
    description: str


# A predictable filler paragraph — repeated to grow the prompt.
_FILLER = (
    "The fleet is humming. Each node serves a tiered queue of model requests. "
    "Routing decisions weigh latency, throughput, and capability. "
    "When a request arrives, the router consults a benchmark matrix and picks "
    "the (model, node) cell that minimizes expected wall time. "
)


def _make_long_prompt(approx_chars: int) -> str:
    reps = max(1, approx_chars // len(_FILLER))
    body = _FILLER * reps
    return (
        "Summarize the following passage in one sentence. Be concise. "
        "Do not add caveats. Passage:\n\n" + body + "\n\nOne-sentence summary:"
    )


SCENARIOS: dict[str, Scenario] = {
    "tiny": Scenario(
        name="tiny",
        prompt="Reply with just the word: ready",
        num_predict=8,
        description="Smallest possible prompt + gen. Pure TTFT + warm cost.",
    ),
    "short": Scenario(
        name="short",
        prompt=(
            "Write a single haiku (5-7-5 syllables) about distributed inference. "
            "Output only the haiku."
        ),
        num_predict=80,
        description="Short prompt, short gen. Common chat-style workload.",
    ),
    "medium": Scenario(
        name="medium",
        prompt=_make_long_prompt(2000),
        num_predict=200,
        description="~500 token prompt, medium gen. Summarization-style.",
    ),
    "long_prompt": Scenario(
        name="long_prompt",
        prompt=_make_long_prompt(8000),
        num_predict=120,
        description="~2k token prompt, short gen. Stresses prompt-eval.",
    ),
    "long_gen": Scenario(
        name="long_gen",
        prompt=(
            "Write a long, detailed technical essay about how a router would "
            "use a benchmark matrix to route LLM inference requests across a "
            "heterogeneous fleet of nodes. Include sections on TTFT, "
            "throughput, queueing, and warmup. Aim for at least 1200 words."
        ),
        num_predict=1200,
        description="Short prompt, long gen. Stresses sustained tok/s.",
    ),
}


def resolve_scenarios(names: list[str]) -> list[Scenario]:
    out: list[Scenario] = []
    for n in names:
        if n not in SCENARIOS:
            raise ValueError(
                f"unknown scenario {n!r}. "
                f"Available: {', '.join(sorted(SCENARIOS))}"
            )
        out.append(SCENARIOS[n])
    return out
