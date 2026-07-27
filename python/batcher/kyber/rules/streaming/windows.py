"""Streaming rule family: windows -- collapsing nested event-time window alignment.

`WindowStart` snaps an instant to the start of the fixed-width tumbling window
containing it, and it is how both a streaming windowed aggregation and a batch
time-bucket query express their grouping key. Stacking two of them is common: a
pipeline buckets to minutes for one stage and the query on top re-buckets to hours,
or a view already windowed at one width is re-windowed by its consumer.

The outer call is redundant exactly when the outer width is a whole multiple of the
inner. Snapping to a 5-minute boundary and then to a 15-minute one lands on the same
instant as snapping straight to 15 minutes, because every 15-minute boundary is also a
5-minute boundary. Width equality is the degenerate case of the same rule -- a value
already on a 5-minute boundary snaps to itself.

The multiple condition is load-bearing, not a convenience. Verified against the engine:
`window(window(t, '5m'), '15m')` and `window(t, '15m')` agree, while
`window(window(t, '5m'), '7m')` yields `04:00` where `window(t, '7m')` yields `04:07`.
Seven is not a multiple of five, so the inner snap moves the instant across an outer
boundary and the two disagree. The origins must match for the same reason: a different
phase offset shifts where the boundaries fall.

This matters more under a stream than in batch. The window start is the grouping key of
a streaming aggregation, so it is computed per row for the life of the query, and it is
what the watermark compares against to decide when a window can be closed and its state
evicted. Removing a redundant snap removes per-row work from an unbounded pipeline.
"""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.kyber.rules.leaf_rewrite import rewrite_node
from batcher.plan.expr_ir import Expr
from batcher.plan.expr_ir.func_nodes import WindowStart
from batcher.plan.logical import Filter, LogicalPlan, Project

__all__ = ["collapse_nested_window_start"]


def _collapse_window_start(expr: Expr) -> Expr:
    if not (isinstance(expr, WindowStart) and isinstance(expr.input, WindowStart)):
        return expr
    inner = expr.input
    if inner.origin_micros != expr.origin_micros:
        return expr  # a different phase moves where the boundaries fall
    if inner.width_micros <= 0 or expr.width_micros % inner.width_micros != 0:
        return expr  # every outer boundary must also be an inner boundary
    return WindowStart(inner.input, expr.width_micros, expr.origin_micros)


@rule(
    name="collapse_nested_window_start",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_collapse_window_start,
)
def collapse_nested_window_start(
    node: Filter | Project, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`window(window(t, 5m), 15m) -> window(t, 15m)`, when 15m is a multiple of 5m.

    Snapping to a boundary and then to a coarser one that contains it lands on the same
    instant as snapping straight to the coarser one, because every coarser boundary is
    also a finer one. Equal widths are the degenerate case: a value already aligned
    snaps to itself.

    Two guards, both verified rather than assumed:

    * **The outer width must be a whole multiple of the inner.** `window(window(t, 5m),
      7m)` gives `04:00` where `window(t, 7m)` gives `04:07` -- the inner snap moves the
      instant back across an outer boundary. Anything that is not a multiple declines.
    * **The origins must match.** A different phase offset puts the boundaries in
      different places, so the containment argument no longer holds.

    Null propagates identically through both forms. Under a streaming query this is
    per-row work removed from a pipeline that never ends, on the very column the
    watermark uses to decide when window state can be evicted."""
    return rewrite_node(node, _collapse_window_start)
