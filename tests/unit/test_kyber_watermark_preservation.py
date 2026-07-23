"""No optimizer rule may silently drop an `Aggregate`'s watermark.

`Aggregate.watermark` is the fourth field and defaults to None, so any rule that rebuilds
the node positionally — `Aggregate(child, keys, aggs)` — drops it. That rewrite looks
harmless and passes every existing test, because the watermark is driver-only: it never
reaches the IR, so the plan's `to_ir()` is byte-identical with or without it.

What it actually does is convert a watermark-bounded streaming aggregate into one whose
state nothing ever evicts. Bounded inputs cannot show this — they release the state at
end-of-input regardless — so the failure appears only on a real stream, as memory growth
over hours. Three separate rules had this bug simultaneously.

Two things are asserted per rule: the watermark survives, and the column it names is
still available underneath (a watermark pointing at a pruned or renamed column is a
watermark that can never advance).
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.kyber.optimizer import Optimizer
from batcher.plan.expr_ir import Col, col, lit
from batcher.plan.logical import Aggregate, AggregateSpec, Filter, Project, Projection
from batcher.plan.streaming import Watermark
from batcher.plan.visitor import walk

pytestmark = pytest.mark.unit


def _watermarked_aggregate(inp, *, time_col="ts", key="k"):
    """A grouped aggregate carrying a watermark on `time_col`."""
    from batcher.plan.expr_ir import AggExpr

    return Aggregate(
        inp,
        group_keys=(Projection(key, Col(key)),),
        aggregates=(AggregateSpec("n", AggExpr("count", Col("v"))),),
        watermark=Watermark(time_col=time_col, lateness_micros=1_000),
    )


def _source():
    return bt.from_pydict({"k": [1, 2], "v": [3, 4], "ts": [10, 20]})._plan


def _aggregates(plan) -> list[Aggregate]:
    return [n for n in walk(plan) if isinstance(n, Aggregate)]


def _rewrite(plan):
    return Optimizer(None, [], None).logical_rewrite(plan)


def test_filter_pushdown_through_aggregate_preserves_the_watermark():
    """`push_filter_through_aggregate` rebuilds the aggregate — it must carry the watermark."""
    agg = _watermarked_aggregate(_source())
    out = _rewrite(Filter(agg, col("k") > lit(0)))
    aggs = _aggregates(out)
    assert aggs, "the aggregate disappeared entirely"
    assert all(a.watermark is not None for a in aggs), "watermark dropped by pushdown"
    assert all(a.watermark.time_col == "ts" for a in aggs)


def test_projection_inlining_into_aggregate_preserves_the_watermark():
    """`projection_inlining_into_agg` re-parents the aggregate past a rename projection."""
    src = _source()
    proj = Project(
        src,
        items=(
            Projection("k", Col("k")),
            Projection("v", Col("v")),
            Projection("ts", Col("ts")),
        ),
    )
    out = _rewrite(_watermarked_aggregate(proj))
    aggs = _aggregates(out)
    assert aggs, "the aggregate disappeared entirely"
    assert all(a.watermark is not None for a in aggs), "watermark dropped by inlining"


def test_projection_inlining_remaps_a_renamed_watermark_column():
    """When the projection renames the event-time column, the watermark must follow it.

    The aggregate is re-parented onto the projection's input, where the column is known
    by its *pre-rename* name. Carrying the watermark forward unchanged would leave it
    naming a column that no longer exists — a watermark that can never advance.
    """
    src = _source()
    proj = Project(
        src,
        items=(
            Projection("k", Col("k")),
            Projection("v", Col("v")),
            Projection("event_ts", Col("ts")),  # ts -> event_ts
        ),
    )
    out = _rewrite(_watermarked_aggregate(proj, time_col="event_ts"))
    for agg in _aggregates(out):
        if agg.watermark is None:
            continue
        available = set(agg.input.available_columns())
        assert agg.watermark.time_col in available, (
            f"watermark names {agg.watermark.time_col!r}, "
            f"but its input only has {sorted(available)}"
        )


def test_column_pruning_keeps_the_watermark_column_alive():
    """Pruning cannot see that the driver reads the event-time column — it must be kept.

    No expression in the plan references `ts`, so ordinary column pruning would remove it
    from the scan. The streaming driver reads it directly to advance the watermark.
    """
    out = _rewrite(_watermarked_aggregate(_source()))
    for agg in _aggregates(out):
        if agg.watermark is None:
            continue
        assert agg.watermark.time_col in set(agg.input.available_columns())


def test_a_watermarkless_aggregate_is_unaffected():
    """The fixes must not invent a watermark where the user asked for none."""
    from batcher.plan.expr_ir import AggExpr

    agg = Aggregate(
        _source(),
        group_keys=(Projection("k", Col("k")),),
        aggregates=(AggregateSpec("n", AggExpr("count", Col("v"))),),
    )
    out = _rewrite(Filter(agg, col("k") > lit(0)))
    assert all(a.watermark is None for a in _aggregates(out))
