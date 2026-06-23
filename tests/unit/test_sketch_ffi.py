"""The newly-surfaced sketch FFI: TDigest, Misra-Gries, reservoir.

These exercise the `bc-py` exports and their Core measure helpers, plus the
`approx_quantile` terminal they back. The underlying sketches' accuracy is tested
in Rust (`bc-sketches`); here we pin the FFI shape and the Python wiring.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytest.importorskip("batcher._native", reason="native engine not built")

from batcher import core


def test_tail_quantiles_helper():
    batches = pa.table({"v": list(range(1000))}).to_batches()
    out = core.tail_quantiles(batches, ["v"], (0.5, 0.99))
    assert "v" in out
    p50, p99 = out["v"]
    assert abs(p50 - 499.5) < 20
    assert p99 > p50 and p99 > 950


def test_tail_quantiles_skips_non_numeric():
    batches = pa.table({"s": ["a", "b", "c"]}).to_batches()
    assert core.tail_quantiles(batches, ["s"], (0.5,)) == {}


def test_heavy_hitters_finds_skew():
    # 'a' is ~60% of rows; with fraction 0.2 it must surface.
    vals = [["a", "a", "a", "b", "c"][i % 5] for i in range(1000)]
    batches = pa.table({"k": vals}).to_batches()
    out = core.heavy_hitters(batches, ["k"], 0.2)
    hits = dict(out["k"])
    assert "a" in hits and hits["a"] > 300


def test_reservoir_sample_size_and_schema():
    batches = pa.table({"x": list(range(500)), "y": list(range(500))}).to_batches()
    import batcher._native as native

    sample = native.reservoir_sample(batches, 20)
    assert sample.num_rows == 20
    assert sample.schema.names == ["x", "y"]


def test_reservoir_sample_returns_all_when_small():
    batches = pa.table({"x": [1, 2, 3]}).to_batches()
    import batcher._native as native

    sample = native.reservoir_sample(batches, 100)
    assert sample.num_rows == 3


def test_approx_quantile_terminal():
    ds = bt.from_pydict({"v": list(range(1000))})
    assert abs(ds.approx_quantile("v", 0.5) - 499.5) < 30
    # Approximate quantile over a filtered result still tracks the data.
    assert ds.filter(col("v") < 100).approx_quantile("v", 0.5) < 100


def test_approx_quantile_non_numeric_is_none():
    ds = bt.from_pydict({"s": ["a", "b", "c"]})
    assert ds.approx_quantile("s", 0.5) is None
