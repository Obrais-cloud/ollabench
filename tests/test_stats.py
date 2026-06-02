from ollabench.stats import percentile, summarize


def test_percentile_empty():
    assert percentile([], 50) == 0.0


def test_percentile_single():
    assert percentile([42.0], 99) == 42.0


def test_percentile_linear_interp():
    vals = [1.0, 2.0, 3.0, 4.0]
    assert percentile(vals, 50) == 2.5
    assert percentile(vals, 0) == 1.0
    assert percentile(vals, 100) == 4.0


def test_summarize_orders():
    s = summarize([5, 1, 4, 2, 3])
    assert s.n == 5
    assert s.min == 1
    assert s.max == 5
    assert s.p50 == 3
    assert round(s.mean, 3) == 3.0


def test_summary_as_dict_roundsafe():
    s = summarize([10.123456, 20.987654])
    d = s.as_dict()
    assert d["n"] == 2
    assert d["min"] == 10.123
    assert d["max"] == 20.988
