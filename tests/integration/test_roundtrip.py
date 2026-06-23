"""End-to-end: api → Kyber → Carbonite → Core → native engine → Arrow.

Requires the compiled engine (`maturin develop`). Skipped cleanly if absent so
pure-Python test runs still pass.
"""

from __future__ import annotations

import pytest

import batcher as bt

pytest.importorskip("batcher._native", reason="native engine not built")


def test_filter_project_roundtrip():
    ds = bt.from_pydict({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50], "name": list("abcde")})
    out = ds.filter(bt.col("x") > 2).select("name", "x", xy=bt.col("x") * bt.col("y")).collect()
    assert out.to_pydict() == {
        "name": ["c", "d", "e"],
        "x": [3, 4, 5],
        "xy": [90, 160, 250],
    }


def test_with_columns_add_and_replace():
    ds = bt.from_pydict({"x": [1, 2, 3]}).with_columns(
        x=bt.col("x") * 10, x2=bt.col("x") * bt.col("x")
    )
    assert ds.collect().to_pydict() == {"x": [10, 20, 30], "x2": [1, 4, 9]}


def test_boolean_filter():
    ds = bt.from_pydict({"a": [1, 2, 3, 4], "b": [1, 0, 1, 0]})
    out = ds.filter((bt.col("a") >= 2) & (bt.col("b") == 1)).select("a", "b").collect()
    assert out.to_pydict() == {"a": [3], "b": [1]}


def test_feedback_recorded_in_metadata_hub():
    from batcher import core

    hub = core.default_hub()
    bt.from_pydict({"x": [1, 2, 3]}).filter(bt.col("x") > 1).collect()
    # Core now records real per-operator feedback (rows in/out, time, peak bytes,
    # backend), keyed by the operator's pre-order id — not one coarse "pipeline" row.
    by_kind = hub.op_stats_by_kind()
    assert "scan" in by_kind, f"expected a scan metric, got {list(by_kind)}"
    assert "filter" in by_kind, f"expected a filter metric, got {list(by_kind)}"
    flt = by_kind["filter"][-1]
    assert flt["n_actual"] == 2  # x > 1 keeps {2, 3}
    assert flt["kind"] != "pipeline"
    # The scan reports a real working-set size — no more hard-coded m_peak_bytes=0.
    assert by_kind["scan"][-1]["m_peak_bytes"] > 0
