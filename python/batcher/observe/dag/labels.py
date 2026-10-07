"""Per-operator labels `explain()` prints beside the plan tree.

What the plan pushed into each scan (filter, row cap, ordering) and what each operator does
in its own terms (join keys, group keys, sort keys). Built from the same
`observe.dag.describe` the dashboard labels its plan nodes with, so the terminal and the web
view cannot drift into showing one operator two ways.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from batcher.observe.dag.describe import describe, expr_text
from batcher.plan.profile import walk_ir

__all__ = ["detail_labels", "pushdown_labels"]


#: How deep a pushed predicate renders. Higher than the node-subtitle default because the
#: optimizer brackets a pushed set with derived bounds, and the interesting part — the
#: column names — sits below the conjunction that adds.
_PUSHED_MAX_DEPTH = 5

#: Longest pushed-filter label. A conjunction of a dozen terms is a real plan and it must
#: not wrap the operator tree it annotates.
_PUSHED_MAX_CHARS = 90


def pushdown_labels(opt) -> dict[int, str]:
    """Per scan `op_id`, what the plan handed that scan's source, for `explain()`.

    `explain()` printed a scan identically whether the plan had pushed a filter into it or
    was reading the whole relation and filtering above — the two differ by the entire
    table, and nothing in the output distinguished them. Every comparable engine says this
    (Spark's ``PushedFilters:``, DuckDB's ``Filters:``), and it is the only way a user can
    confirm that a filter they expected to prune actually reached the source.

    Read as *offered*, not *applied*: `PhysicalPlan.source_predicates` is what the plan
    hands down, and each backend then translates the subset it can express (see
    `io.predicate`). A source that declines still shows the offer here, which is the
    honest report — the alternative is asking every source what it did, which it cannot
    answer until it runs. The same holds for the row cap (`source_limits`), which most
    backends decline: a database is only sent a ``LIMIT`` when its dialect spells one.

    The pushed *projection* is deliberately not shown. Whether a column list is pruning
    anything is only knowable against the source's own schema, and reading that here would
    put a probe round trip on the execution path to label a line — while the columns the
    query keeps are already on the `project` node directly above.

    Args:
        opt: The `PhysicalPlan`, carrying `source_predicates` and `source_limits` per
            scan `source_id`.

    Returns:
        A mapping from scan `op_id` to its label; scans the plan pushed nothing to are
        absent.
    """
    labels: dict[int, str] = {}
    for op_id, (_depth, node) in enumerate(walk_ir(opt.ir)):
        if node.get("op") != "scan":
            continue
        source_id = node.get("source_id")
        parts = []
        predicate = opt.source_predicates.get(source_id)
        if predicate:
            parts.append(_elide(expr_text(predicate, max_depth=_PUSHED_MAX_DEPTH)))
        cap = opt.source_limits.get(source_id)
        if cap is not None:
            ordering = opt.source_orderings.get(source_id)
            parts.append(
                f"top {cap:,} by {_ordering_text(ordering)}" if ordering else f"max {cap:,} rows"
            )
        if parts:
            labels[op_id] = " · ".join(parts)
    return labels


def _ordering_text(ordering: tuple[tuple[str, bool, bool], ...]) -> str:
    """A pushed top-N's sort keys, for the scan's label.

    Shown because the ordering is what makes the cap *sound*: "max 2 rows" under a sort
    reads like an unsound prefix, and naming the order it is taken in is the difference.
    Null placement is printed only when it is not the default, so the common case stays
    short while the case that changes which rows come back stays visible.
    """
    return ", ".join(
        f"{column}{' desc' if descending else ''}{' nulls first' if nulls_first else ''}"
        for column, descending, nulls_first in ordering
    )


def _elide(text: str) -> str:
    """`text` cut to one readable line."""
    if len(text) <= _PUSHED_MAX_CHARS:
        return text
    return text[: _PUSHED_MAX_CHARS - 1].rstrip() + "…"


def detail_labels(ir: dict | None, sources: Sequence[Any] = ()) -> dict[int, str]:
    """Per `op_id`, what that operator does in its own terms, for `explain()`.

    The join type and keys, the group keys and aggregates, the sort keys, the filter
    predicate. `explain()` printed none of it, so a plan with four joins printed four
    identical `hash_join` lines and the reader had no way to tell which was which — the
    first question anyone asks of a join tree. Every comparable engine prints it
    (Postgres's ``Hash Cond:``, Spark's ``[id#3 = id#7]``, DuckDB's key list).

    Reuses `observe.dag.describe`, which is the same function the web dashboard labels its
    plan nodes with, rather than growing a second describer: two of them would drift within
    a release and show the same operator two ways, which is the failure that makes a reader
    stop trusting both.

    A scan whose source names the backend it was routed to (a database read's
    `explain_label`, such as ``dbapi(sqlite3)``) carries that too, since routing picks it
    from the URI and what is installed and nothing else in the plan says which.

    Args:
        ir: The optimized plan IR, walked in the pre-order that assigns `op_id`.
        sources: The plan's sources, indexed by a scan's `source_id`.

    Returns:
        A mapping from `op_id` to its label; operators with nothing worth naming are absent.
    """
    if not ir:
        return {}
    labels: dict[int, str] = {}
    for op_id, (_depth, node) in enumerate(walk_ir(ir)):
        text = describe(str(node.get("op", "")), node)
        if node.get("op") == "scan":
            text = " · ".join(filter(None, (text, _source_label(sources, node))))
        if text:
            labels[op_id] = _elide(text)
    return labels


def _source_label(sources: Sequence[Any], node: dict) -> str:
    """The backend a scan's source was routed to, or `""` when it does not say."""
    source_id = node.get("source_id", 0)
    if not isinstance(source_id, int) or not 0 <= source_id < len(sources):
        return ""
    label = getattr(sources[source_id], "explain_label", None)
    return label() if callable(label) else ""
