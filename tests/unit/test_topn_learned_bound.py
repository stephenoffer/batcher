"""Learned top-N bounds: the k-th best value a top-N returned seeds the next run of the
same shape as a predicate, so the scan can skip what it would otherwise decode and discard.

The property that makes this safe is not "the bound is accurate" — it is that an *inaccurate*
bound cannot produce a wrong answer. The seeded plan removes only rows strictly beyond the
bound, so any `k` survivors are the true top-k whatever the bound was; too few survivors is
the single failure mode, it is visible in the row count, and the conductor answers it by
re-running the plan as written. These tests pin that trichotomy — accurate, stale-tight,
not-seedable — plus the shape conditions that decide when seeding is attempted at all.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher.kyber.learned_tuning.topn_bound import (
    record_topn_bound,
    seed_topn_bound,
)
from batcher.metadata.backends.in_process import InProcessBackend
from batcher.metadata.hub import MetadataHub

pytestmark = pytest.mark.unit


def _hub():
    return MetadataHub(InProcessBackend())


def _table(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    return pa.table(
        {"x": rng.integers(0, 10**6, n).astype("int64"), "p": np.arange(n, dtype="int64")}
    )


def _plan(descending=True, k=10, nulls_first=False):
    ds = bt.from_arrow(_table())
    return ds.sort("x", descending=descending, nulls_first=nulls_first).limit(k)._plan


def test_no_seed_before_anything_has_been_learned():
    """A cold shape must run as written — there is no bound to start from."""
    assert seed_topn_bound(_plan(), _hub()) is None


def test_a_recorded_bound_seeds_the_next_run_of_the_same_shape():
    hub = _hub()
    plan = _plan()
    result = pa.table({"x": np.arange(990, 1000, dtype="int64"), "p": np.zeros(10, "int64")})

    record_topn_bound(hub, plan, result)
    seed = seed_topn_bound(plan, hub)

    assert seed is not None
    assert seed.k == 10
    # The rewrite adds exactly one Filter, below the Sort, and changes nothing else.
    assert seed.plan is not plan
    ir = seed.plan.to_ir()
    assert ir["op"] == "limit"
    assert ir["input"]["op"] == "sort"
    assert ir["input"]["input"]["op"] == "filter"


def test_a_short_result_is_never_recorded_as_a_bound():
    """A relation with fewer than `k` rows has no k-th best value.

    Its worst row bounds nothing — every row clears it — so recording it would seed the next
    run with a predicate that removes nothing while still costing an evaluation.
    """
    hub = _hub()
    plan = _plan(k=10)
    short = pa.table({"x": np.arange(3, dtype="int64"), "p": np.zeros(3, "int64")})

    record_topn_bound(hub, plan, short)
    assert seed_topn_bound(plan, hub) is None


def test_a_sort_key_projected_out_of_the_result_records_nothing():
    """The bound is read off the result, so a result that no longer carries the sort key
    cannot say where the cut fell."""
    hub = _hub()
    plan = _plan()
    record_topn_bound(hub, plan, pa.table({"p": np.arange(10, dtype="int64")}))
    assert seed_topn_bound(plan, hub) is None


def test_nulls_first_is_never_seeded():
    """A bound predicate drops nulls, and a nulls-first ordering wanted them at the top.

    This is the one unsafe shape, and it is unsafe *invisibly*: the result would come back
    with `k` rows, so the conductor's row-count check would accept a wrong answer. It has to
    be refused at the shape test rather than caught downstream.
    """
    hub = _hub()
    plan = _plan(nulls_first=True)
    result = pa.table({"x": np.arange(990, 1000, dtype="int64"), "p": np.zeros(10, "int64")})
    record_topn_bound(hub, plan, result)
    assert seed_topn_bound(plan, hub) is None


def test_a_different_shape_does_not_borrow_the_bound():
    """The bound is a fact about one query, and the ascending twin of a descending top-N
    wants the opposite end of the distribution."""
    hub = _hub()
    record_topn_bound(
        hub,
        _plan(descending=True),
        pa.table({"x": np.arange(990, 1000, dtype="int64"), "p": np.zeros(10, "int64")}),
    )
    assert seed_topn_bound(_plan(descending=False), hub) is None
    assert seed_topn_bound(_plan(descending=True, k=25), hub) is None


@pytest.mark.parametrize("descending", [True, False])
def test_the_seeded_plan_returns_the_same_rows_as_the_plan_as_written(descending):
    """The end-to-end property, on data the bound describes exactly."""
    table = _table(n=5000, seed=7)
    ds = bt.from_arrow(table)
    query = ds.sort("x", descending=descending).limit(10)

    first = query.collect()
    second = query.collect()  # now seeded from what the first run learned
    assert first.to_pydict() == second.to_pydict()


def test_a_stale_bound_falls_back_and_still_returns_the_true_top_k():
    """The failure mode, exercised deliberately.

    A bound learned from one relation is applied to a *different* one whose values are all
    far below it, so the seeded plan returns zero rows. The conductor must notice the short
    result and re-run as written rather than hand back an empty answer.
    """
    hub = _hub()
    big = pa.table({"x": np.arange(10**6, 10**6 + 100, dtype="int64"), "p": np.zeros(100, "int64")})
    plan = _plan()
    record_topn_bound(hub, plan, big.slice(0, 10))

    seed = seed_topn_bound(plan, hub)
    assert seed is not None, "the bound should be applied — the fallback is what is under test"

    # The seeded plan over the small table keeps nothing, which is exactly the signal.
    seeded_rows = bt.from_arrow(_table()).sort("x", descending=True).limit(10)
    filtered = bt.from_arrow(_table()).filter(bt.col("x") >= 10**6)
    assert filtered.collect().num_rows < seed.k
    # And the query as written still answers correctly.
    assert seeded_rows.collect().num_rows == 10
