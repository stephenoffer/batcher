"""Session windows: grouping a partition's events into runs separated by an idle gap.

Layer: `api` (surface). Split out of `core.py` because a session window is the one
construct here that is not sugar over an existing verb — it needs a per-partition scan
to decide where a session *starts*, and the bounded and unbounded inputs take different
routes to it (`StreamingSessionWindow` against a lag-and-cumulative-sum rewrite).

Everything still lowers to existing IR; nothing here adds a node beyond the streaming
operator the unbounded path already had.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.api.dataset._build.core import _all_bounded, build_window

if TYPE_CHECKING:
    from batcher.api.dataset.frame import Dataset


def build_session_window(
    ds: Dataset, time_col: str, gap: str, partition_by: list[str], aggs: dict[str, Any]
) -> Dataset:
    """Session-window aggregation (Spark ``session_window``) with no new operator.

    A session groups consecutive events (per `partition_by`) whose gap is below
    `gap`. Computed by composing existing ops: order by event time, mark a *new
    session* where the gap to the previous event exceeds `gap` (or it is the first),
    assign a session id as the running sum of those markers, then group by
    ``partition_by + session_id`` — emitting ``session_start``/``session_end`` and the
    requested aggregates. All row work is the existing window + group-by engine.
    """
    from batcher.plan.functions.temporal import _duration_micros

    if time_col not in ds.columns:
        raise PlanError(f"session_window(): unknown time column {time_col!r}")
    # Checked here rather than left to the window node underneath: by then the plan
    # carries the internal `_t` epoch copy, so the "available columns" the error listed
    # included a column the caller never wrote and cannot use.
    unknown = [name for name in partition_by if name not in ds.columns]
    if unknown:
        raise PlanError(
            f"session_window(): unknown partition_by column(s) {unknown}; "
            f"available: {sorted(ds.columns)}"
        )
    if not aggs:
        raise PlanError("session_window() requires at least one named aggregate")
    gap_us = _duration_micros(gap, arg="session gap")
    pk = list(partition_by)

    # Over an unbounded source the composition below cannot run: a session's end is not
    # known until the gap has elapsed with nothing arriving, so the operator has to hold
    # rows until the watermark says so. That is a driver-side stateful node, and it
    # re-applies these same aggregates to each closed batch.
    if not _all_bounded(ds):
        from batcher.plan.logical import StreamingSessionWindow

        lateness = ds._watermark.lateness_micros if ds._watermark is not None else 0
        return ds._derive(
            StreamingSessionWindow(
                ds._plan, time_col, gap_us, tuple(pk), tuple(aggs.items()), lateness
            )
        )
    return sessionize(ds, time_col, gap_us, pk, aggs)


def mark_sessions(ds: Dataset, time_col: str, gap_us: int, pk: list[str]) -> Dataset:
    """Tag every row with the session it belongs to, as `_t`/`_sid`.

    Ordered by event time within each `pk` group, a row starts a new session when the gap
    to the row before it exceeds `gap_us` (or there is no row before it); the session id
    is the running count of those starts. `_t` is the event time in epoch microseconds,
    which the gap arithmetic and the session bounds both need — the engine has no
    Timestamp min/max.

    Args:
        ds: The rows to tag.
        time_col: The event-time column.
        gap_us: The gap that separates two sessions, in microseconds.
        pk: The partition-key columns; empty for one global session chain.

    Returns:
        `ds` with `_t` and `_sid` added.
    """
    from batcher.plan.expr_ir import col

    order = [(time_col, False)]
    # Through `timestamp` first, not straight to int64. A `date32` column casts to int64 as
    # a count of *days*, which was then compared against a gap expressed in microseconds --
    # so every gap looked smaller than every threshold and a whole key collapsed into one
    # session. Silently: the answer was one plausible row per key.
    s = ds.with_columns(_t=col(time_col).cast("timestamp").cast("int64"))
    s = build_window(
        s, partition_by=pk, order_by=order, functions={"_prev": ("lag", "_t", 1)}, frame=None
    )
    new_session = ((col("_t") - col("_prev") > gap_us) | col("_prev").is_null()).cast("int64")
    s = s.with_columns(_new=new_session)
    return build_window(
        s, partition_by=pk, order_by=order, functions={"_sid": ("sum", "_new")}, frame=None
    )


def sessionize(
    ds: Dataset, time_col: str, gap_us: int, pk: list[str], aggs: dict[str, Any]
) -> Dataset:
    """Aggregate `ds` by gap-based session, emitting `session_start`/`session_end`.

    The bounded computation, and the one the streaming operator runs over each batch of
    rows whose sessions the watermark has closed — so the two paths agree by construction
    rather than by a second implementation agreeing with the first.

    Args:
        ds: The rows to sessionize.
        time_col: The event-time column.
        gap_us: The gap that separates two sessions, in microseconds.
        pk: The partition-key columns.
        aggs: The named aggregates to compute per session.

    Returns:
        One row per session: the partition keys, its bounds, and the aggregates.
    """
    from batcher.plan.expr_ir import col

    s = mark_sessions(ds, time_col, gap_us, pk)
    # The bounds are min/max of the event-time column *itself*, not of the epoch-micros
    # copy the gap arithmetic needs. Going through the copy and casting back produced a
    # naive `timestamp[us]` whatever the input was, so a `timestamp[us, tz=UTC]` column
    # came back with the right instant and no timezone, and a `timestamp[ms]` column came
    # back in microseconds. Right values, wrong type — which anything rendering a local
    # time downstream reads as a wrong answer.
    grouped = s.group_by(*pk, "_sid").agg(
        session_start=col(time_col).min(), session_end=col(time_col).max(), **aggs
    )
    return grouped.select(*pk, "session_start", "session_end", *aggs.keys())
