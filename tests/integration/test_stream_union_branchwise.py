"""UNION ALL streams branch by branch instead of materializing the concatenation.

`api/terminal/stream/dispatch.py` had no `Union` branch, so `a.union(b).iter_batches()`
fell through to `_collect` and materialized the whole concatenation — the one thing a
caller reaching for `iter_batches` is trying to avoid — and a union touching an unbounded
source raised rather than streaming.

A UNION ALL's result *is* its branches' results concatenated, so yielding each branch's
own stream in order is bounded in memory and identical row-for-row and in order. The three
preconditions that make that true are each pinned below, because skipping any of them is a
wrong answer rather than a slow one: UNION (distinct) needs a global dedup, an unbounded
branch would never yield control to branch k+1, and a branch pair whose types differ is
owed a promotion this path does not perform.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt


def _rows(ds) -> list[dict]:
    batches = list(ds.iter_batches())
    return pa.Table.from_batches(batches).to_pylist() if batches else []


@pytest.fixture
def a():
    return bt.from_pydict({"x": list(range(2_000)), "s": [f"a{i}" for i in range(2_000)]})


@pytest.fixture
def b():
    return bt.from_pydict({"x": list(range(2_000, 4_000)), "s": [f"b{i}" for i in range(2_000)]})


@pytest.mark.integration
@pytest.mark.parametrize(
    "shape",
    [
        lambda a, b: a.union(b),
        lambda a, b: a.union(b).union(a),
        lambda a, b: a.filter(bt.col("x") > 100).union(b.filter(bt.col("x") < 3_000)),
        lambda a, b: a.union(b).select("x"),
        lambda a, b: (
            a.group_by("x").agg(t=bt.col("x").sum()).union(b.group_by("x").agg(t=bt.col("x").sum()))
        ),
        lambda a, b: a.sort("x", descending=True).union(b.sort("x")),
    ],
    ids=["plain", "three-branch", "filtered", "projected-above", "agg-branches", "sort-branches"],
)
def test_streamed_union_equals_collected(a, b, shape):
    ds = shape(a, b)
    # Order-DEPENDENT on purpose. Concatenation order is the whole contract of this path
    # (branch 0's rows then branch 1's), and two of these shapes sort — an order-independent
    # comparison could not see either bug (CLAUDE.md's explicit warning).
    assert _rows(ds) == ds.collect().to_pylist()


@pytest.mark.integration
def test_union_all_no_longer_materializes(a, b, monkeypatch):
    """The point of the change: `_collect` is not reached, and the first batch arrives
    without the second branch having been read."""
    import batcher.api.terminal.core as tc

    calls: list[int] = []
    original = tc._collect
    monkeypatch.setattr(tc, "_collect", lambda *ar, **k: (calls.append(1), original(*ar, **k))[1])

    it = a.union(b).iter_batches()
    first = next(it)
    assert first.num_rows > 0
    assert calls == []
    assert first.num_rows + sum(x.num_rows for x in it) == 4_000
    assert calls == []


@pytest.mark.integration
def test_union_distinct_still_dedups(a):
    """UNION (distinct) must NOT take the branch-wise path — it needs a global dedup."""
    ds = a.union(a, distinct=True)
    streamed = _rows(ds)
    assert len(streamed) == 2_000
    assert streamed == ds.collect().to_pylist()


@pytest.mark.integration
def test_union_of_mismatched_types_still_promotes(a):
    """A branch pair whose types differ is owed the engine's promotion, so it must fall
    back to the materialized path rather than yield each branch's own narrower type."""
    narrow = bt.from_pydict({"x": pa.array([1, 2, 3], pa.int32()), "s": ["p", "q", "r"]})
    ds = a.union(narrow)
    batches = list(ds.iter_batches())
    assert {batch.schema.field("x").type for batch in batches} == {pa.int64()}
    assert _rows(ds) == ds.collect().to_pylist()


@pytest.mark.integration
def test_union_batch_size_contract_holds(a, b):
    """`batch_size` is an exact output-granularity contract; only the final batch is short."""
    sizes = [batch.num_rows for batch in a.union(b).iter_batches(batch_size=700)]
    assert sizes[:-1] == [700] * (len(sizes) - 1)
    assert sum(sizes) == 4_000
