"""`ds.count()` answered by the source's own ``COUNT(*)`` instead of by reading it.

The sibling modules here answer a terminal from statistics already in hand, for free. This
one costs exactly one round trip, and is separated for that reason — it is consulted only
after the free answers decline, and only on the `count()` terminal itself.

That distinction is the whole design. `Source.row_count` is asked while *planning*, where
Kyber wants a cheap estimate and has a better one in the learned statistics; a warehouse
source answers `None` there on purpose, because charging a query per plan would be a bad
trade. But `ds.count()` is not a plan-time estimate. When the free answers decline, the
fallback is to wrap the plan in a `COUNT(*)` aggregate and *execute* it — and a
``COUNT(*)`` needs no columns, so the projection that would have narrowed the read is
empty, which the SQL builder renders as ``SELECT *``. Counting a warehouse table therefore
pulled every column of every row across the network to produce one integer.

Only the shapes where the source's own count *is* the answer are pushed: a bare scan, or
projections over one. A projection cannot change how many rows there are. Anything else —
a filter, a join, an aggregate, a limit — declines and the ordinary path runs, exactly as
before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.logging import note_suppressed

if TYPE_CHECKING:
    from batcher.io.source import Source
    from batcher.plan.logical import LogicalPlan

__all__ = ["pushed_count"]


def pushed_count(plan: LogicalPlan, sources: list[Source]) -> int | None:
    """The row count computed by the source itself, or None to run the plan.

    Args:
        plan: The user's plan, as written.
        sources: The plan's bound sources, indexed by `Scan.source_id`.

    Returns:
        The exact row count, or None when this plan or this source cannot answer.
    """
    source = _countable_source(plan, sources)
    if source is None:
        return None
    try:
        return source.exact_row_count()
    except Exception as exc:
        # Counting is an optimization: any failure falls back to running the plan, which
        # is what every source that cannot count does anyway.
        note_suppressed("api", "answer count() from the source", exc)
        return None


def _countable_source(plan: LogicalPlan, sources: list[Source]) -> Source | None:
    """The single source whose own row count answers `plan`, or None."""
    from batcher.plan.logical import Project, Scan

    node = plan
    while isinstance(node, Project):
        node = node.input
    if not isinstance(node, Scan):
        return None
    if not 0 <= node.source_id < len(sources):
        return None
    source = sources[node.source_id]
    if not getattr(source, "supports_count", False):
        return None
    return source
