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


def test_distributed_adaptive_aggregate_then_join_equals_single_node():
    # An aggregate feeding a join: the aggregate is an *intermediate* breaker, so the
    # distributed adaptive path keeps its result partitioned on disk (a
    # MaterializedSource) and scans it in place for the join — never collecting it to
    # the driver. The result must still equal single-node.
    fact, dim = _big_small()

    def query(ds_fact, ds_dim):
        agg = ds_fact.group_by("k").agg(s=col("v").sum())
        return agg.join(ds_dim, on="k")

    base = query(fact, dim).collect()
    adaptive_dist = query(fact, dim).collect(distributed=True, adaptive=True, num_workers=2)
    assert _rowset(adaptive_dist) == _rowset(base)


def test_execute_distributed_materialize_false_keeps_aggregate_partitioned():
    # The mechanism: a distributed aggregate run with materialize=False returns a
    # MaterializedSource over its on-disk reducer output (not a collected table), with
    # an exact row count and the same data as the collected path.
    from batcher.dist import execute_distributed
    from batcher.io.source import MaterializedSource

    fact, _ = _big_small()
    agg = fact.group_by("k").agg(s=col("v").sum())
    plan, sources = agg._plan, agg._sources

    collected = execute_distributed(plan, sources, num_workers=2, transport="disk")
    partitioned = execute_distributed(
        plan, sources, num_workers=2, transport="disk", materialize=False
    )
    try:
        assert isinstance(partitioned, MaterializedSource)
        assert partitioned.row_count() == collected.num_rows
        scanned = pa.Table.from_batches(partitioned.read(), schema=partitioned.schema())
        assert _rowset(scanned) == _rowset(collected)
    finally:
        partitioned.cleanup()


def test_flight_aggregate_materialize_false_keeps_result_on_actors():
    # The multi-node mechanism: a Flight aggregate run with materialize=False keeps its
    # result on the worker actors and returns a FlightMaterializedSource (not a
    # collected table) — its buckets are read back shared-nothing over Flight. The
    # actors persist until cleanup(); the data + exact count match the collected path.
    # (Adaptive cross-stage Flight materialize is deferred — it needs a shared actor
    # fleet to avoid placement contention — so this exercises the mechanism directly.)
    from batcher.dist.flight_aggregate import execute_aggregate_flight
    from batcher.dist.flight_worker import FlightMaterializedSource

    fact, _ = _big_small()
    agg = fact.group_by("k").agg(s=col("v").sum())
    plan, sources = agg._plan, agg._sources

    collected = execute_aggregate_flight([], plan, sources, 2)
    partitioned = execute_aggregate_flight([], plan, sources, 2, materialize=False)
    try:
        assert isinstance(partitioned, FlightMaterializedSource)
        assert partitioned.row_count() == collected.num_rows
        scanned = pa.Table.from_batches(partitioned.read(), schema=partitioned.schema())
        assert _rowset(scanned) == _rowset(collected)
    finally:
        partitioned.cleanup()


def test_execute_distributed_materialize_false_keeps_join_partitioned():
    # The same mechanism for a co-partition shuffle join: materialize=False returns a
    # MaterializedSource over the reducer IPC output (not a collected table) with an
    # exact row count and identical data — the common join → group-by adaptive pattern.
    from batcher.dist import execute_distributed
    from batcher.io.source import MaterializedSource

    fact, dim = _big_small()
    joined = fact.join(dim, on="k")
    plan, sources = joined._plan, joined._sources

    collected = execute_distributed(plan, sources, num_workers=2, transport="disk")
    partitioned = execute_distributed(
        plan, sources, num_workers=2, transport="disk", materialize=False
    )
    try:
        assert isinstance(partitioned, MaterializedSource)
        assert partitioned.row_count() == collected.num_rows
        scanned = pa.Table.from_batches(partitioned.read(), schema=partitioned.schema())
        assert _rowset(scanned) == _rowset(collected)
    finally:
        partitioned.cleanup()
