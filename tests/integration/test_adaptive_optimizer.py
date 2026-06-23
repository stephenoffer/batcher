"""Cardinality-driven adaptive optimization + cross-execution learning."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import Config, col, config_context
from batcher.config.config import MetadataConfig

pytest.importorskip("batcher._native", reason="native engine not built")


@pytest.fixture(autouse=True)
def _isolate_metadata_hub():
    """Reset the process-wide MetadataHub around each test.

    These tests assert on cardinality/cost-driven plan shape (build-side swaps,
    learned selectivity), which the shared hub's accumulated stats from earlier
    tests would otherwise perturb — the source of cross-run flakiness.
    """
    from batcher.core import reset_default_hub

    reset_default_hub()
    yield
    reset_default_hub()


def _big_small():
    fact = bt.from_arrow(pa.table({"k": np.arange(100_000) % 50, "v": np.arange(100_000) % 7}))
    dim = bt.from_arrow(pa.table({"k": np.arange(50), "label": [f"d{i}" for i in range(50)]}))
    return fact, dim


def test_build_side_keeps_small_on_right():
    fact, dim = _big_small()
    # fact (big) on left, dim (small) on right → small already builds; no swap.
    assert "keep" in fact.join(dim, on="k").explain()


def test_build_side_swaps_to_build_smaller():
    fact, dim = _big_small()
    # dim (small) on left, fact (big) on right → swap so small is the build side.
    assert "SWAP" in dim.join(fact, on="k").explain()


def test_build_side_swap_preserves_results():
    fact, dim = _big_small()
    a = fact.join(dim, on="k").group_by("label").agg(s=col("v").sum()).collect()
    b = dim.join(fact, on="k").group_by("label").agg(s=col("v").sum()).collect()

    def rowset(t):
        c = t.column_names
        return sorted(tuple(r[k] for k in c) for r in t.to_pylist())

    assert rowset(a) == rowset(b)


def test_cross_execution_learning_refines_selectivity(tmp_path):
    uri = str(tmp_path / "stats.db")
    with config_context(Config().replace(metadata=MetadataConfig(backend="sqlite", uri=uri))):
        n = 100_000
        t = pa.table({"x": (np.arange(n) % 100).astype("int64"), "v": np.arange(n) % 5})

        def q():
            return bt.from_arrow(t).filter(col("x") < 1)  # ~1% selective

        before = q().explain()
        assert "default" in before  # no knowledge yet

        actual = q().count()  # executes → records measured size
        assert actual < n // 50  # genuinely selective

        after = q().explain()
        assert "learned" in after  # estimate now reflects the measured size


def test_estimate_distinct_native():
    import batcher._native as nat

    batch = pa.record_batch({"x": pa.array([i % 1000 for i in range(50_000)], pa.int64())})
    ndv = nat.estimate_distinct("x", [batch])
    assert abs(ndv - 1000) / 1000 < 0.05  # HLL++ within ~5%


def _rowset(t):
    c = t.column_names
    return sorted(tuple(r[k] for k in c) for r in t.to_pylist())


def test_distributed_adaptive_equals_single_node():
    # The moat: adaptive re-optimization now also runs distributed. A multi-stage
    # query (join feeding a group-by) must produce identical results whether run
    # single-node, single-node+adaptive, or distributed+adaptive — the mergeable
    # algebra + measured-cardinality re-planning never change the relation.
    fact, dim = _big_small()

    def query(ds_fact, ds_dim):
        return ds_fact.join(ds_dim, on="k").group_by("label").agg(s=col("v").sum())

    base = query(fact, dim).collect()
    adaptive_local = query(fact, dim).collect(adaptive=True)
    adaptive_dist = query(fact, dim).collect(distributed=True, adaptive=True, num_workers=2)

    assert _rowset(adaptive_local) == _rowset(base)
    assert _rowset(adaptive_dist) == _rowset(base)
