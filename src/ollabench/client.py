"""Async Ollama client with TTFT-aware streaming.

Records wall-clock TTFT (time-to-first-token), per-chunk arrival times, and the
final `done` payload from Ollama so the bench can derive prompt-eval and
generation rates separately.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class NodeConfig:
    name: str
    host: str
    tags: list[str] = field(default_factory=list)


@dataclass
class GenerateSample:
    """One full streaming /api/generate run with timing breakdown."""

    response: str
    eval_count: int
    prompt_eval_count: int

    # Wall-clock measurements (what the client actually saw).
    ttft_ms: float | None
    total_ms: float

    # Server-reported breakdown (nanoseconds → ms).
    load_ms: float
    prompt_eval_ms: float
    eval_ms: float

    # Derived rates.
    gen_tok_s: float          # eval_count / eval_duration
    wall_tok_s: float         # eval_count / (total - ttft)

    error: str | None = None


class OllamaNode:
    def __init__(self, config: NodeConfig, timeout: float = 600.0):
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=config.host,
            timeout=httpx.Timeout(timeout, connect=5.0),
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def ping(self) -> tuple[bool, str | None]:
        try:
            resp = await self.client.get("/api/version", timeout=5.0)
            resp.raise_for_status()
            return True, resp.json().get("version")
        except Exception as e:
            return False, str(e)

    async def list_models(self) -> list[str]:
        try:
            resp = await self.client.get("/api/tags", timeout=10.0)
            resp.raise_for_status()
            return [m.get("name", "") for m in resp.json().get("models", [])]
        except Exception:
            return []

    async def has_model(self, model: str) -> bool:
        names = await self.list_models()
        return any(model == n or model in n for n in names)

    async def generate(
        self,
        model: str,
        prompt: str,
        *,
        num_predict: int,
        options: dict | None = None,
    ) -> GenerateSample:
        opts = dict(options or {})
        opts.setdefault("num_predict", num_predict)

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": opts,
        }

        start = time.monotonic()
        ttft_ms: float | None = None
        chunks: list[str] = []
        final: dict = {}

        try:
            async with self.client.stream("POST", "/api/generate", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    piece = data.get("response", "")
                    if piece:
                        if ttft_ms is None:
                            ttft_ms = (time.monotonic() - start) * 1000
                        chunks.append(piece)
                    if data.get("done"):
                        final = data
        except Exception as e:
            total = (time.monotonic() - start) * 1000
            return GenerateSample(
                response="",
                eval_count=0,
                prompt_eval_count=0,
                ttft_ms=ttft_ms,
                total_ms=total,
                load_ms=0.0,
                prompt_eval_ms=0.0,
                eval_ms=0.0,
                gen_tok_s=0.0,
                wall_tok_s=0.0,
                error=f"{type(e).__name__}: {e}",
            )

        total_ms = (time.monotonic() - start) * 1000
        eval_count = int(final.get("eval_count", 0))
        prompt_eval_count = int(final.get("prompt_eval_count", 0))
        eval_dur_ns = int(final.get("eval_duration", 0))
        prompt_dur_ns = int(final.get("prompt_eval_duration", 0))
        load_ns = int(final.get("load_duration", 0))

        eval_ms = eval_dur_ns / 1e6
        prompt_eval_ms = prompt_dur_ns / 1e6
        load_ms = load_ns / 1e6

        gen_tok_s = (eval_count / (eval_dur_ns / 1e9)) if eval_dur_ns > 0 else 0.0
        wall_gen_ms = total_ms - (ttft_ms or 0.0)
        wall_tok_s = (eval_count / (wall_gen_ms / 1000.0)) if wall_gen_ms > 0 else 0.0

        return GenerateSample(
            response="".join(chunks),
            eval_count=eval_count,
            prompt_eval_count=prompt_eval_count,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            load_ms=load_ms,
            prompt_eval_ms=prompt_eval_ms,
            eval_ms=eval_ms,
            gen_tok_s=gen_tok_s,
            wall_tok_s=wall_tok_s,
        )
