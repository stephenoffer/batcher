"""Kyber's streaming rules and the streaming analysis they gate on.

Plan-shape assertions live here; result correctness against DuckDB lives in
`tests/differential/`. Each rule gets a pair — one test that it fires, one that it
declines on the case where firing would be wrong — because for these rules the
declining case is the one that protects a correct answer.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.streaming import (
    blocking_operators,
    emits_incrementally,
    has_unbounded_input,
    is_blocking_under_stream,
    retains_unbounded_state,
)
from batcher.plan.expr_ir import col, lit
from batcher.plan.logical import (
    Aggregate,
    Distinct,
    Filter,
    Sort,
    WatermarkDedup,
)
from batcher.plan.visitor import walk

pytestmark = pytest.mark.unit


def _source():
    return bt.from_pydict({"k": [1, 2, 3], "v": [10, 20, 30], "t": [1, 2, 3]})


def _dedup(plan, subset=("k",)):
    return WatermarkDedup(plan, subset=subset, event_time="t", lateness_micros=1_000)


def _shape(plan) -> list[str]:
    return [type(n).__name__ for n in walk(plan)]


def _rewrite(plan):
    return Optimizer(None, [], None).logical_rewrite(plan)


# --- push_filter_through_watermark_dedup ---------------------------------------


def test_key_constant_filter_pushes_below_watermark_dedup():
    """A predicate over the dedup `subset` may cross it — and shrinks the seen-key state."""
    plan = Filter(_dedup(_source()._plan), col("k") > lit(0))
    shape = _shape(_rewrite(plan))
    assert shape.index("WatermarkDedup") < shape.index("Filter"), shape


def test_non_key_filter_does_not_push_below_watermark_dedup():
    """A predicate over a non-key column must NOT cross the dedup.

    The dedup keeps the *first* row per key. A predicate that accepts some rows of a key
    and rejects others would, if applied first, promote a different row to be that key's
    first — changing the output. Nothing about that failure is loud, so it is pinned here.
    """
    plan = Filter(_dedup(_source()._plan), col("v") > lit(0))
    shape = _shape(_rewrite(plan))
    assert shape.index("Filter") < shape.index("WatermarkDedup"), shape


def test_event_time_filter_does_not_push_below_watermark_dedup():
    """The event-time column is not key-constant — two rows of a key differ in it."""
    plan = Filter(_dedup(_source()._plan), col("t") > lit(0))
    shape = _shape(_rewrite(plan))
    assert shape.index("Filter") < shape.index("WatermarkDedup"), shape


def test_optimizing_a_streaming_node_preserves_its_fields():
    """The rewrite must not disturb the dedup's state-bearing configuration."""
    original = _dedup(_source()._plan)
    out = _rewrite(Filter(original, col("k") > lit(0)))
    assert isinstance(out, WatermarkDedup)
    assert out.subset == original.subset
    assert out.event_time == original.event_time
    assert out.lateness_micros == original.lateness_micros


# --- the streaming analysis ----------------------------------------------------


def test_full_sort_blocks_under_a_stream_but_topn_does_not():
    """A top-N keeps a bounded running best-N, so it can emit; a full sort cannot."""
    assert is_blocking_under_stream(Sort(_source()._plan, keys=(), limit=None))
    assert not is_blocking_under_stream(Sort(_source()._plan, keys=(), limit=10))


def test_distinct_blocks_under_a_stream():
    assert is_blocking_under_stream(Distinct(_source()._plan))


def test_grouped_aggregate_does_not_block():
    """A grouped aggregate emits a running result per group — the streamable case."""
    agg = _source().group_by("k").agg(s=col("v").sum())._plan
    assert isinstance(agg, Aggregate)
    assert not is_blocking_under_stream(agg)
    assert blocking_operators(agg) == []


def test_grouped_aggregate_without_a_watermark_retains_unbounded_state():
    """Correct, and a memory leak: nothing ever evicts a group without a watermark."""
    agg = _source().group_by("k").agg(s=col("v").sum())._plan
    assert retains_unbounded_state(agg)


def test_bounded_plan_is_never_reported_unbounded():
    """`has_unbounded_input` reads the bound source, not a row estimate.

    An unknown row count means "not estimated" — which a finite source reports just as
    readily as a stream — so boundedness must not be inferred from cardinality.
    """
    ctx = Optimizer(None, [], None)._context()
    plan = _source()._plan
    assert not has_unbounded_input(plan, ctx)
    assert emits_incrementally(plan, ctx)
