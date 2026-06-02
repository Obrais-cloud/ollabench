import json
from pathlib import Path

from ollabench.compare import diff, filter_significant


def _write(tmp: Path, matrix: dict) -> Path:
    p = tmp / f"{matrix['_name']}.json"
    p.write_text(json.dumps({"meta": {}, "matrix": matrix["data"]}))
    return p


def test_diff_detects_regression(tmp_path):
    a = _write(tmp_path, {
        "_name": "a",
        "data": {
            "m1|n1|short": {"ttft_p50_ms": 100.0, "gen_tok_s_p50": 50.0, "total_p50_ms": 1000.0, "ttft_p95_ms": 150.0},
        },
    })
    b = _write(tmp_path, {
        "_name": "b",
        "data": {
            "m1|n1|short": {"ttft_p50_ms": 200.0, "gen_tok_s_p50": 50.0, "total_p50_ms": 1000.0, "ttft_p95_ms": 150.0},
        },
    })
    deltas = diff(a, b)
    ttft = next(d for d in deltas if d.metric == "ttft_p50_ms")
    assert ttft.pct == 100.0
    assert ttft.regressed


def test_diff_detects_improvement_for_tok_s(tmp_path):
    a = _write(tmp_path, {
        "_name": "a",
        "data": {
            "m1|n1|short": {"ttft_p50_ms": 100.0, "gen_tok_s_p50": 50.0, "total_p50_ms": 1000.0, "ttft_p95_ms": 150.0},
        },
    })
    b = _write(tmp_path, {
        "_name": "b",
        "data": {
            "m1|n1|short": {"ttft_p50_ms": 100.0, "gen_tok_s_p50": 75.0, "total_p50_ms": 1000.0, "ttft_p95_ms": 150.0},
        },
    })
    deltas = diff(a, b)
    tok = next(d for d in deltas if d.metric == "gen_tok_s_p50")
    assert tok.pct == 50.0
    assert tok.improved
    assert not tok.regressed


def test_filter_significant_threshold(tmp_path):
    a = _write(tmp_path, {
        "_name": "a",
        "data": {
            "m1|n1|short": {"ttft_p50_ms": 100.0, "gen_tok_s_p50": 50.0, "total_p50_ms": 1000.0, "ttft_p95_ms": 150.0},
        },
    })
    b = _write(tmp_path, {
        "_name": "b",
        "data": {
            "m1|n1|short": {"ttft_p50_ms": 105.0, "gen_tok_s_p50": 50.0, "total_p50_ms": 1000.0, "ttft_p95_ms": 150.0},
        },
    })
    deltas = diff(a, b)
    assert filter_significant(deltas, 10.0) == []
    assert filter_significant(deltas, 1.0)
