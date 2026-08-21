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


# --- The contract, not the three rules that broke it --------------------------------------
#
# The tests above each name one rule. That is why the bug kept coming back: the *next* rule to
# rebuild an `Aggregate` positionally was covered by none of them, and there turned out to be
# fifteen such sites across five files. This asserts the property instead -- for a spread of
# aggregate shapes, whatever the optimizer does to them, a watermark that went in comes out.
#
# Two shapes here are not "carry the field" but "decline the rewrite": `count_distinct` and an
# aggregate whose only function folds into a group key both rewrote into a `Distinct`, which
# has **no** watermark field and so cannot carry one at all. A rewrite that cannot represent a
# field has to decline rather than drop it.

#: name -> a builder over a watermarked streaming Dataset.
_WATERMARKED_SHAPES = {
    "plain_sum": lambda d: d.group_by("k").agg(s=bt.col("v").sum()),
    "multi_key": lambda d: d.group_by("k", "s").agg(tot=bt.col("v").sum()),
    "global": lambda d: d.agg(s=bt.col("v").sum()),
    "constant_group_key": lambda d: d.group_by("k", "c").agg(s=bt.col("v").sum()),
    "duplicate_aggregates": lambda d: d.group_by("k").agg(a=bt.col("v").sum(), b=bt.col("v").sum()),
    "count_of_a_constant": lambda d: d.group_by("k").agg(n=bt.lit(1).count()),
    "sum_of_a_constant": lambda d: d.group_by("k").agg(n=bt.lit(1).sum()),
    "over_a_filter": lambda d: d.filter(bt.col("v") > 1).group_by("k").agg(s=bt.col("v").sum()),
    "over_a_sort": lambda d: d.sort("v").group_by("k").agg(s=bt.col("v").sum()),
    "over_a_distinct": lambda d: d.distinct().group_by("k").agg(s=bt.col("v").sum()),
    # These two rewrote into a `Distinct` and so dropped the watermark structurally.
    "aggregate_of_a_group_key": lambda d: d.group_by("k").agg(m=bt.col("k").max()),
    "count_of_a_group_key": lambda d: d.group_by("k").agg(n=bt.col("k").count()),
    "count_distinct": lambda d: d.group_by("k").agg(n=bt.col("k").n_unique()),
}


def _streaming_frame():
    return bt.from_pydict(
        {"k": [1, 1, 2], "v": [3, 4, 5], "c": [9, 9, 9], "s": ["a", "b", "b"], "ts": [10, 20, 30]}
    ).with_watermark("ts", "10m")


@pytest.mark.parametrize("shape", sorted(_WATERMARKED_SHAPES))
def test_a_watermark_survives_optimization_whatever_the_aggregate_shape(shape):
    from batcher import core, kyber

    ds = _WATERMARKED_SHAPES[shape](_streaming_frame())
    assert any(a.watermark is not None for a in _aggregates(ds._plan)), (
        "the shape must carry a watermark before optimization, or it proves nothing"
    )
    optimized = kyber.optimize_logical(ds._plan, sources=ds._sources, hub=core.default_hub())
    carriers = [n for n in walk(optimized) if getattr(n, "watermark", None) is not None]
    assert carriers, (
        f"{shape}: the watermark was dropped by optimization. Either the rule that rebuilt the "
        f"node must carry it (use `dataclasses.replace`), or -- if it rewrites into a node with "
        f"no watermark field -- it must decline while one is set."
    )


@pytest.mark.parametrize("shape", sorted(_WATERMARKED_SHAPES))
def test_the_same_shapes_still_optimize_when_no_watermark_is_set(shape):
    """The declines above are conditional: a bounded query keeps every rewrite it had."""
    from batcher import core, kyber

    ds = _WATERMARKED_SHAPES[shape](
        bt.from_pydict(
            {
                "k": [1, 1, 2],
                "v": [3, 4, 5],
                "c": [9, 9, 9],
                "s": ["a", "b", "b"],
                "ts": [10, 20, 30],
            }
        )
    )
    optimized = kyber.optimize_logical(ds._plan, sources=ds._sources, hub=core.default_hub())
    assert not [n for n in walk(optimized) if getattr(n, "watermark", None) is not None]
    assert ds.collect().num_rows > 0
