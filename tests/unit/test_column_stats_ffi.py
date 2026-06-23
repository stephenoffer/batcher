"""The `column_stats` FFI seam — per-column optimizer statistics from the engine.

This is the W2 metadata FFI keystone: Core can collect these per-column summaries
(HLL distinct + KLL quantiles, mergeable across batches) and persist them so Kyber's
cardinality estimator can populate `__column_ndv__` and range selectivity.
"""

from __future__ import annotations

import batcher._native as _native
import pyarrow as pa


def test_column_stats_numeric_and_string():
    t = pa.table(
        {
            "x": list(range(1000)),  # 1000 distinct ints, no nulls
            "y": (["a", "b", "c"] * 333) + ["a"],  # 3 distinct strings
            "n": ([None] * 500) + list(range(500)),  # half null
        }
    )
    batches = t.to_batches(max_chunksize=137)  # exercise the cross-batch merge
    stats = _native.column_stats(["x", "y", "n"], batches)

    # x — numeric: exact min/max, ~1000 distinct, no nulls.
    assert abs(stats["x"]["ndv"] - 1000) / 1000 < 0.05
    assert stats["x"]["min"] == 0.0
    assert stats["x"]["max"] == 999.0
    assert stats["x"]["count"] == 1000.0
    assert stats["x"]["null_fraction"] == 0.0

    # y — string: distinct works on any type; no quantile sketch → min/max None.
    assert abs(stats["y"]["ndv"] - 3.0) < 1.0
    assert stats["y"]["min"] is None
    assert stats["y"]["max"] is None

    # n — half null.
    assert stats["n"]["count"] == 1000.0
    assert stats["n"]["null_count"] == 500.0
    assert stats["n"]["null_fraction"] == 0.5


def test_column_stats_missing_column_omitted():
    t = pa.table({"a": [1, 2, 3]})
    stats = _native.column_stats(["a", "nope"], t.to_batches())
    assert "a" in stats
    assert "nope" not in stats  # absent columns are simply skipped


def test_column_stats_feeds_cardinality_ndv_hook():
    # End-to-end: the FFI ndv populates the estimator's __column_ndv__ hook, so
    # equality selectivity sharpens to ~1/ndv instead of the 0.1 default.
    import batcher as bt
    from batcher import col, lit
    from batcher.kyber.cardinality import CardinalityEstimator

    ds = bt.from_pydict({"k": [i % 50 for i in range(1000)]})
    batches = pa.table({"k": [i % 50 for i in range(1000)]}).to_batches()
    ndv = {name: s["ndv"] for name, s in _native.column_stats(["k"], batches).items()}

    flt = ds.filter(col("k") == lit(7))
    est = CardinalityEstimator(flt._sources, learned={"__column_ndv__": ndv})
    # ndv(k) ≈ 50 → selectivity ≈ 1/50 → ~20 rows (vs 100 with the 0.1 default).
    assert est.estimate(flt._plan).rows < 40


def test_column_quantiles_numeric_and_string():
    t = pa.table({"v": list(range(1000)), "s": ["a", "b", "c"] * 333 + ["a"]})
    probs = [0.0, 0.25, 0.5, 0.75, 1.0]
    q = _native.column_quantiles(["v", "s"], t.to_batches(max_chunksize=128), probs)
    vq = q["v"]
    assert len(vq) == 5
    assert vq == sorted(vq)  # ascending boundaries
    assert vq[0] <= 5 and vq[-1] >= 995  # spans the range
    assert abs(vq[2] - 500) < 75  # median near 500 (KLL approximate)
    assert q["s"] == []  # non-numeric → empty


def test_range_selectivity_from_quantiles():
    # Uniform v in 0..999: `v < 250` keeps ~25%, sharper than the 1/3 default; the
    # FFI quantiles populate the estimator's __column_quantiles__ histogram hook.
    import batcher as bt
    from batcher import col, lit
    from batcher.kyber.cardinality import CardinalityEstimator

    vals = list(range(1000))
    ds = bt.from_pydict({"v": vals})
    probs = [i / 10 for i in range(11)]
    qv = _native.column_quantiles(["v"], pa.table({"v": vals}).to_batches(), probs)["v"]
    learned = {"__column_quantiles__": {"v": {"probs": probs, "values": qv}}}

    below = ds.filter(col("v") < lit(250))
    est = CardinalityEstimator(below._sources, learned=learned)
    rows = est.estimate(below._plan).rows
    assert 180 < rows < 320  # ~250, and clearly below the 333 (1/3) default

    above = ds.filter(col("v") > lit(750))
    est2 = CardinalityEstimator(above._sources, learned=learned)
    assert 180 < est2.estimate(above._plan).rows < 320  # ~250 (complement)

    # Without the hook, the same range falls back to the 1/3 Selinger constant.
    base = CardinalityEstimator(below._sources)
    assert abs(base.estimate(below._plan).rows - 1000 / 3) < 1
