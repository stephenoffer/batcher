"""Carbonite fault tolerance end to end: a lost worker's shuffle is recomputed.

Kills a worker after the map barrier (so its published buckets vanish) and asserts
the distributed aggregate still equals the single-node result — Spark-style lineage
recompute, driven by `ShuffleRecovery`, produces the correct answer despite the
worker loss.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, count

pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("batcher._native", reason="native engine not built")


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    import ray

    ray.init(num_cpus=4, include_dashboard=False, logging_level="ERROR", ignore_reinit_error=True)
    yield
    ray.shutdown()


def _data():
    rng = np.random.default_rng(19)
    n = 120_000
    return pa.table(
        {"k": rng.integers(0, 40, n).astype("int64"), "v": rng.integers(0, 100, n).astype("int64")}
    )


def _norm(t: pa.Table) -> set:
    return {
        tuple(round(v, 6) if isinstance(v, float) else v for v in r.values()) for r in t.to_pylist()
    }


@pytest.mark.parametrize("killed", [{1}, {0, 2}])
def test_aggregate_survives_worker_loss(killed):
    from batcher.dist.flight_aggregate import execute_aggregate_flight

    t = _data()
    expected = bt.from_arrow(t).group_by("k").agg(s=col("v").sum(), n=count()).collect()

    ds = bt.from_arrow(t).group_by("k").agg(s=col("v").sum(), n=count())
    recovered = execute_aggregate_flight([], ds._plan, ds._sources, workers=4, _fault_inject=killed)

    assert _norm(recovered) == _norm(expected)


def test_no_fault_is_unaffected():
    from batcher.dist.flight_aggregate import execute_aggregate_flight

    t = _data()
    expected = bt.from_arrow(t).group_by("k").agg(s=col("v").sum()).collect()
    ds = bt.from_arrow(t).group_by("k").agg(s=col("v").sum())
    got = execute_aggregate_flight([], ds._plan, ds._sources, workers=4)
    assert _norm(got) == _norm(expected)


@pytest.mark.parametrize("killed", [{1}, {0, 2}])
def test_window_survives_worker_loss(killed):
    """A lost worker's window row-bucket is recomputed from its source on a survivor;
    the Flight window shuffle still equals single-node despite the loss."""
    from batcher.dist.flight_window import execute_window_flight

    t = _data()
    expected = bt.from_arrow(t).window(partition_by=["k"], functions={"s": ("sum", "v")}).collect()

    ds = bt.from_arrow(t).window(partition_by=["k"], functions={"s": ("sum", "v")})
    recovered = execute_window_flight([], ds._plan, ds._sources, workers=4, _fault_inject=killed)

    assert _norm(recovered) == _norm(expected)


@pytest.mark.parametrize("killed", [{1}, {0, 2}])
def test_sort_survives_worker_loss(killed):
    """A lost worker's range bucket is recomputed from its source on a survivor; the
    Flight range-sort still produces the globally sorted result despite the loss."""
    from batcher.dist.flight_sort import execute_sort_flight

    t = _data()
    expected = bt.from_arrow(t).sort("k").collect()

    ds = bt.from_arrow(t).sort("k")
    recovered = execute_sort_flight([], ds._plan, ds._sources, workers=4, _fault_inject=killed)

    assert recovered.column("k").to_pylist() == expected.column("k").to_pylist()
