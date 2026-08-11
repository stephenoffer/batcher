"""The Flight transport's broadcast join and its skew salting, against single-node.

Both are *scheduling* changes that must not move a row. The broadcast path replicates a
small build side and shuffles nothing; salting spreads a hot key's rows over several
reducers. Each is engaged deliberately here and its result compared to the single-node
answer, because both are reachable by default on a real cluster and neither raises when it
is wrong — a broadcast join that mis-handled an outer join's null extension, or a salted
shuffle whose build rows failed to follow their probe rows, returns a quietly short answer.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from _ray_cluster import init_test_ray, shutdown_test_ray

pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("batcher._native", reason="native engine not built")

WORKERS = 4


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    started = init_test_ray(WORKERS)
    yield
    shutdown_test_ray(started)


def _rowset(t: pa.Table) -> set:
    cols = t.column_names
    return {tuple(r[c] for c in cols) for r in t.to_pylist()}


def _sides(*, nulls: bool = False, hot: bool = False):
    """A probe side and a small build side. `nulls` puts unmatched and NULL keys on the
    probe (the shapes an outer join is about); `hot` puts 40% of the probe on one key."""
    rng = np.random.default_rng(5)
    n = 40_000
    if hot:
        keys = np.concatenate(
            [np.zeros(n * 2 // 5, dtype="int64"), rng.integers(0, 60, n - n * 2 // 5)]
        )
        rng.shuffle(keys)
    else:
        keys = rng.integers(0, 80, n).astype("int64")
    kcol: pa.Array = pa.array(keys, pa.int64())
    if nulls:
        # Unmatched (>= 60, absent from the build) and NULL keys, which never match at all.
        vals = [
            None if i % 97 == 0 else int(k) + (500 if i % 89 == 0 else 0)
            for i, k in enumerate(keys)
        ]
        kcol = pa.array(vals, pa.int64())
    left = pa.table({"k": kcol, "lv": pa.array(rng.integers(0, 50, n), pa.int64())})
    right = pa.table(
        {"k": pa.array(np.arange(60), pa.int64()), "label": [f"g{i}" for i in range(60)]}
    )
    return left, right


def _plan(left, right, how):
    return bt.from_arrow(left).join(bt.from_arrow(right), on="k", how=how)


@pytest.mark.integration
@pytest.mark.parametrize("how", ["inner", "left", "semi", "anti"])
@pytest.mark.parametrize("nulls", [False, True])
def test_broadcast_join_matches_single_node(how, nulls):
    """The replicated-build-side path returns the single-node relation for every join type
    it claims. `left`/`anti` with NULL and unmatched probe keys are the sharp cases: those
    rows have no build partner, so they exist in the output only if each worker emits its
    own unmatched rows — which is sound precisely because a probe row lands on exactly one
    worker and sees the whole build side there."""
    from batcher.dist.flight_broadcast import execute_broadcast_join_flight

    left, right = _sides(nulls=nulls)
    ds = _plan(left, right, how)
    expected = ds.collect()
    # `strategy` is the planner's; set it here so the path under test is the one that runs
    # rather than whatever the current threshold happens to decide.
    plan = dataclasses.replace(ds._plan, strategy="broadcast")
    got = execute_broadcast_join_flight([], plan, ds._sources, workers=WORKERS)
    assert got is not None, "a 60-row build side must be inside the broadcast budget"
    assert _rowset(expected) == _rowset(got)


@pytest.mark.integration
def test_broadcast_join_declines_an_oversized_build_side():
    """The measured guard, not the estimate. A build side over the budget must return None
    so the caller falls back to the co-partition shuffle — the planner decides on an
    estimate, and replicating a mis-estimated side to every worker is the one failure this
    strategy has that is worse than being slow."""
    from batcher.config import active_config, set_config
    from batcher.dist.flight_broadcast import execute_broadcast_join_flight

    left, right = _sides()
    ds = _plan(left, right, "inner")
    plan = dataclasses.replace(ds._plan, strategy="broadcast")
    cfg = active_config()
    set_config(cfg.replace(optimizer=dataclasses.replace(cfg.optimizer, broadcast_max_bytes=1)))
    try:
        assert execute_broadcast_join_flight([], plan, ds._sources, workers=WORKERS) is None
    finally:
        set_config(cfg)


@pytest.mark.integration
def test_broadcast_join_with_an_empty_build_side_falls_back():
    """An empty build side is handed back to the shuffle rather than answered here, so the
    outer-join null extension comes from the one path that builds it from real schemas."""
    from batcher.dist.flight_broadcast import execute_broadcast_join_flight

    left, _ = _sides()
    empty = pa.table({"k": pa.array([], pa.int64()), "label": pa.array([], pa.string())})
    ds = _plan(left, empty, "left")
    plan = dataclasses.replace(ds._plan, strategy="broadcast")
    assert execute_broadcast_join_flight([], plan, ds._sources, workers=WORKERS) is None


@pytest.mark.integration
@pytest.mark.parametrize("how", ["inner", "left"])
def test_salted_shuffle_join_matches_unsalted(how):
    """Salting a measured hot key moves work between reducers and nothing else.

    Driven through `execute_join_flight` with the detection pre-pass forced on, against the
    same join run unsalted, so the comparison is of the two schedules rather than of a
    schedule against a different query. The hot key holds 40% of the probe side, which is
    what makes its build rows have to reach every one of its sub-buckets.
    """
    from batcher.config import active_config, set_config
    from batcher.dist.flight_join import execute_join_flight

    left, right = _sides(hot=True)
    ds = _plan(left, right, how)
    expected = ds.collect()
    cfg = active_config()
    set_config(cfg.replace(distributed=dataclasses.replace(cfg.distributed, skew_join_salt=8)))
    try:
        salted = execute_join_flight([], ds._plan, ds._sources, workers=WORKERS)
    finally:
        set_config(cfg)
    assert _rowset(expected) == _rowset(salted)


@pytest.mark.integration
def test_salting_is_withheld_from_a_finalizing_reducer():
    """A reducer that finalizes a fused aggregate relies on co-partitioning putting each
    group in one bucket — exactly what salting breaks. The guard is what keeps the fused
    join+aggregate correct, and its failure mode is a silently split group rather than an
    error, so it is pinned rather than trusted."""
    from batcher.dist.skew import salting_preserves_result

    left, right = _sides(hot=True)
    join = _plan(left, right, "inner")._plan
    assert salting_preserves_result(join, reducer_finalizes=False)
    assert not salting_preserves_result(join, reducer_finalizes=True)
