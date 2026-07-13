"""Dispatch a MERGE to the target format's own implementation, when it has one.

A transactional table's client can perform a real ``MERGE INTO``: it joins the change set
against the table, rewrites only the data files the join touches, and commits them as one
version. That is strictly better than the copy-on-write path — atomic, concurrent-safe, and
sublinear in the table — and for a lakehouse target it is also the *only* thing that works,
since the copy-on-write path rebuilds its target from an explicit file list and a lakehouse
source cannot be constructed that way.

This module is the one place that knows which formats have a native merge and how to reach
it. Both entry points funnel through it — the `merge` shorthand and the `merge_into`
builder — which is the point: they used to disagree. The shorthand short-circuited Delta
into a hard-coded ``update_all`` + ``insert_all`` (silently ignoring a ``delete`` clause),
and the builder consulted the native set at all, so `merge_into` fell into the file path
and crashed on the very format with a native MERGE.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset
    from batcher.api.merge.clauses import MergeClause
    from batcher.io.manifest import WriteManifest

__all__ = ["NATIVE_MERGE_SINKS", "merge_predicate_for", "native_merge"]

#: Formats whose own client performs a real MERGE. The single source of truth — `Writer`
#: used to keep a private copy and the builder consulted neither.
NATIVE_MERGE_SINKS = frozenset({"delta", "iceberg"})


def native_merge(
    source: Dataset,
    target: str,
    keys: list[str],
    clauses: Sequence[MergeClause],
    fmt: str,
    opts: dict,
) -> WriteManifest:
    """Run `clauses` through the native MERGE of the format at `target`.

    The change set is materialized, because a merge *is* a join against it — but a change
    set is a delta rather than a bulk load, so the cost is bounded by the size of the
    update and not of the table.

    Args:
        source: The change set.
        target: The table being merged into.
        keys: The columns matching a source row to a target row.
        clauses: The ordered ``WHEN`` clauses.
        fmt: The target's format; must be in `NATIVE_MERGE_SINKS`.
        opts: Sink options (``storage_options`` for Delta, ``catalog`` for Iceberg).

    Returns:
        A `WriteManifest` of the data files the merge wrote.

    Raises:
        PlanError: If `fmt` has no native merge, or the clauses use a shape it cannot run.
    """
    if fmt == "delta":
        from batcher.api.merge.delta_native import merge_into_delta

        return merge_into_delta(
            target,
            source.collect(),
            keys,
            list(clauses),
            storage_options=opts.get("storage_options"),
        )
    if fmt == "iceberg":
        from batcher.api.merge.iceberg_native import merge_into_iceberg

        return merge_into_iceberg(target, source.collect(), keys, list(clauses), opts)
    raise PlanError(f"merge(): {fmt!r} declares a native merge but none is wired")


def merge_predicate_for(keys: str | list[str]) -> str:
    """Build a Delta ``MERGE`` match predicate from key column(s).

    ``merge_predicate_for(["id", "day"])`` →
    ``"target.id = source.id AND target.day = source.day"`` — the engine's delta
    sink aliases the existing table ``target`` and the new data ``source``.
    """
    cols = [keys] if isinstance(keys, str) else list(keys)
    if not cols:
        raise PlanError("merge_on requires at least one key column")
    return " AND ".join(f"target.{c} = source.{c}" for c in cols)
