"""Distributed join over Flight: correctness and worker-loss recovery.

Brings the join path to parity with the aggregate path — a lost worker's
co-partitioned left and right buckets are recomputed from their source partitions
on a survivor, so the join completes and equals the single-node result.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from _ray_cluster import init_test_ray, shutdown_test_ray

pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("batcher._native", reason="native engine not built")


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    started = init_test_ray(4)
    yield
    shutdown_test_ray(started)


def _tables():
    rng = np.random.default_rng(13)
    n = 80_000
    left = pa.table(
        {"k": rng.integers(0, 80, n).astype("int64"), "lv": rng.integers(0, 50, n).astype("int64")}
    )
    right = pa.table({"k": np.arange(80, dtype="int64"), "label": [f"g{i}" for i in range(80)]})
    return left, right


def _rowset(t: pa.Table) -> set:
    cols = t.column_names
    return {tuple(r[c] for c in cols) for r in t.to_pylist()}


def test_join_flight_matches_single_node():
    from batcher.dist.flight_join import execute_join_flight

    left, right = _tables()
    expected = bt.from_arrow(left).join(bt.from_arrow(right), on="k", how="inner").collect()
    ds = bt.from_arrow(left).join(bt.from_arrow(right), on="k", how="inner")
    got = execute_join_flight([], ds._plan, ds._sources, workers=4)
    assert _rowset(expected) == _rowset(got)


@pytest.mark.parametrize("killed", [{2}, {1, 3}])
def test_join_flight_survives_worker_loss(killed):
    from batcher.dist.flight_join import execute_join_flight

    left, right = _tables()
    expected = bt.from_arrow(left).join(bt.from_arrow(right), on="k", how="inner").collect()
    ds = bt.from_arrow(left).join(bt.from_arrow(right), on="k", how="inner")
    recovered = execute_join_flight([], ds._plan, ds._sources, workers=4, _fault_inject=killed)
    assert _rowset(expected) == _rowset(recovered)


@pytest.mark.parametrize("killed", [{2}, {1, 3}])
def test_join_flight_survives_worker_loss_during_map(killed):
    """A worker preempted *during* the map barrier has BOTH its co-partitioned sides
    republished on one survivor under the same source id, so the reducers' tickets still
    resolve and the join equals single-node. The barrier used to be a bare `ray.get`, so
    a preemption in the map phase failed the whole join."""
    from batcher.dist.flight_join import execute_join_flight

    left, right = _tables()
    expected = bt.from_arrow(left).join(bt.from_arrow(right), on="k", how="inner").collect()
    ds = bt.from_arrow(left).join(bt.from_arrow(right), on="k", how="inner")
    recovered = execute_join_flight([], ds._plan, ds._sources, workers=4, _fault_inject_map=killed)
    assert _rowset(expected) == _rowset(recovered)
