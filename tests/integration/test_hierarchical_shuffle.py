"""Hierarchical bounded-fan-in tree shuffle equals single-node.

When a shuffle has more upstreams than the fan-in bound, the reduce runs as a tree
of combiner stages so no node ever reads from more than `shuffle_fan_in` upstreams.
Forcing `fan_in=2` with 4 workers builds a 2-level tree; the aggregate must still
equal the single-node result. This is the property that bounds per-node fan-in as
the cluster grows to many thousands of nodes.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, count
from batcher.config import Config, FlowControlConfig, config_context

pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("batcher._native", reason="native engine not built")


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    import ray

    ray.init(num_cpus=4, include_dashboard=False, logging_level="ERROR", ignore_reinit_error=True)
    yield
    ray.shutdown()


def _data():
    rng = np.random.default_rng(23)
    n = 120_000
    return pa.table(
        {"k": rng.integers(0, 60, n).astype("int64"), "v": rng.integers(0, 100, n).astype("int64")}
    )


def _norm(t: pa.Table) -> set:
    return {
        tuple(round(v, 6) if isinstance(v, float) else v for v in r.values()) for r in t.to_pylist()
    }


def _tree_cfg(fan_in: int):
    return Config().replace(flow_control=FlowControlConfig(shuffle_fan_in=fan_in))


def test_tree_grouped_aggregate_matches_single_node():
    t = _data()

    def q(ds):
        return ds.group_by("k").agg(
            s=col("v").sum(), n=count(), a=col("v").mean(), hi=col("v").max()
        )

    single = q(bt.from_arrow(t)).collect()
    with config_context(_tree_cfg(2)):  # 4 workers > fan_in 2 → 2-level combiner tree
        tree = q(bt.from_arrow(t)).collect(distributed=True, num_workers=4)
    assert _norm(single) == _norm(tree)


def test_tree_global_aggregate_matches_single_node():
    t = _data()

    def q(ds):
        return ds.group_by().agg(s=col("v").sum(), n=count(), a=col("v").mean())

    single = q(bt.from_arrow(t)).collect().to_pydict()
    with config_context(_tree_cfg(2)):
        tree = q(bt.from_arrow(t)).collect(distributed=True, num_workers=4).to_pydict()
    assert single == tree


@pytest.mark.parametrize("killed", [{1}, {0, 3}])
def test_tree_shuffle_survives_worker_loss(killed):
    """The bounded-fan-in tree path also recovers from worker loss: a lost worker's
    leaf partition is recomputed on a survivor and the tree is rebuilt — so the
    large-cluster path is fault-tolerant, not just scalable."""
    from batcher.dist.flight_aggregate import execute_aggregate_flight

    t = _data()
    expected = bt.from_arrow(t).group_by("k").agg(s=col("v").sum(), n=count()).collect()
    ds = bt.from_arrow(t).group_by("k").agg(s=col("v").sum(), n=count())
    with config_context(_tree_cfg(2)):  # 4 workers > fan_in 2 → tree path
        recovered = execute_aggregate_flight(
            [], ds._plan, ds._sources, workers=4, _fault_inject=killed
        )
    assert _norm(expected) == _norm(recovered)


def test_fan_in_threshold_keeps_small_shuffle_flat():
    # workers <= fan_in uses the flat reduce (still correct) — the tree only kicks
    # in above the bound. With the default fan_in this is the common path.
    t = _data()

    def q(ds):
        return ds.group_by("k").agg(s=col("v").sum())

    single = q(bt.from_arrow(t)).collect()
    with config_context(_tree_cfg(8)):  # 4 workers <= 8 → flat path
        flat = q(bt.from_arrow(t)).collect(distributed=True, num_workers=4)
    assert _norm(single) == _norm(flat)
