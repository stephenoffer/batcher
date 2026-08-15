"""How many rows each source may stop after — the row-cap half of source pushdown.

`required_columns_per_source` tells a source which columns to read and
`required_predicates_per_source` tells it which rows to skip. Neither tells it when to
*stop*, so ``bt.read.postgres(...).limit(100)`` issued an unbounded ``SELECT`` and pulled
the whole table across the network to discard all but a hundred rows. The same shape on a
file source decoded every row group. Spark, DuckDB and Daft all push a limit; this is the
analysis that lets Batcher.

**A `LIMIT` is a positional prefix, so almost nothing may sit between it and the scan.**
`Filter` is the case that looks safe and is not: ``Limit(Filter(p, x), n)`` takes the first
`n` rows *that pass* `p`, while capping the source at `n` takes the passing rows of the
first `n` — fewer rows, or none at all if the first `n` all fail. The same argument rules
out `Sort` (which reads everything before it can know the first row), `Aggregate`,
`Distinct`, `Sample`, `Unnest` and `MapBatches`. Only `Project` passes a prefix through
unchanged: it is row-for-row and order-preserving, so the `n`th projected row is the `n`th
scanned row.

Two properties make the cap safe once extracted:

* **`offset` is added, not subtracted.** ``Limit(n, offset)`` needs the source to supply
  ``offset + n`` rows, because the engine skips the first `offset` of them itself.
* **A cap is a ceiling, never a floor.** A source free to return more rows than asked is
  still correct — the engine's own `Limit` truncates — so a source that ignores the cap
  behaves exactly as it does today. That is what makes this an optimization rather than a
  semantic change, and it is why a source is only capped when *every* scan of it is capped:
  one unbounded scan of a source means no cap at all, exactly as an unfiltered scan blocks
  a pushed predicate in `required_predicates_per_source`.
"""

from __future__ import annotations

from batcher.plan.expr_ir import Col
from batcher.plan.logical import Limit, LogicalPlan, Project, Scan, Sort
from batcher.plan.visitor import children

__all__ = ["required_limits_per_source", "required_orderings_per_source"]


#: What a scan may be told: how many rows, and — for a top-N — in what order.
_Cap = tuple[int, tuple[tuple[str, bool, bool], ...] | None]


def required_limits_per_source(plan: LogicalPlan) -> dict[int, int]:
    """Per scan `source_id`, the most rows that source ever needs to produce.

    Args:
        plan: The optimized logical plan.

    Returns:
        A mapping from `source_id` to a row cap. A source absent from the mapping has at
        least one scan that reads it unbounded and must not be capped.
    """
    return {source_id: cap for source_id, (cap, _order) in _caps(plan).items()}


def required_orderings_per_source(
    plan: LogicalPlan,
) -> dict[int, tuple[tuple[str, bool, bool], ...]]:
    """Per scan `source_id`, the ordering its row cap is taken in, for a top-N.

    Only ever populated alongside `required_limits_per_source`. Ordering a source without
    capping it is a pessimization — the server sorts rows it would otherwise have streamed
    — so an ordering is pushed exactly when it is what *makes* a cap sound.

    Args:
        plan: The optimized logical plan.

    Returns:
        A mapping from `source_id` to its sort keys, each ``(column, descending,
        nulls_first)``. Absent when the scan's cap is an unordered prefix.
    """
    return {
        source_id: order for source_id, (_cap, order) in _caps(plan).items() if order is not None
    }


def _caps(plan: LogicalPlan) -> dict[int, _Cap]:
    """Per scan `source_id`, the reconciled cap and ordering, or nothing."""
    caps: dict[int, _Cap | None] = {}
    _collect(plan, None, caps)
    return {source_id: cap for source_id, cap in caps.items() if cap is not None}


def _sort_columns(node: Sort) -> tuple[tuple[str, bool, bool], ...] | None:
    """`node`'s keys as plain column names, or None if any key is computed.

    A server can only order by something it can name. An expression key would have to be
    re-derived in the pushed SQL, and a mistranslation there would not read extra rows —
    it would return a *different* top-N, so the whole ordering declines instead.
    """
    keys = []
    for key in node.keys:
        if not isinstance(key.expr, Col):
            return None
        keys.append((key.expr.name, bool(key.descending), bool(key.nulls_first)))
    return tuple(keys) or None


def _collect(node: LogicalPlan, pending: _Cap | None, caps: dict[int, _Cap | None]) -> None:
    """Walk down carrying the cap an enclosing `Limit` or top-N imposes on this subtree.

    `pending` is what the subtree beneath `node` still has to be able to produce — a row
    count, and for a top-N the ordering that count is taken in — or None where no cap
    applies. It survives only the operators that preserve a positional prefix, and is
    dropped at every other one.
    """
    if isinstance(node, Scan):
        # `None` wins over any cap and is never overwritten: it records that some scan of
        # this source reads it unbounded, which no other scan's cap may narrow.
        if node.source_id in caps and caps[node.source_id] is None:
            return
        if pending is None:
            caps[node.source_id] = None
            return
        seen = caps.get(node.source_id)
        if seen is None:
            caps[node.source_id] = pending
        elif seen[1] != pending[1]:
            # Two scans of one source wanting different orderings: no single read serves
            # both, and picking either would starve the other.
            caps[node.source_id] = None
        else:
            caps[node.source_id] = (max(seen[0], pending[0]), seen[1])
        return
    if isinstance(node, Limit):
        # The engine skips `offset` rows itself, so the source must supply them too.
        # Tightened against an enclosing cap rather than replacing it: stacked limits are
        # normally folded by `combine_limits`, but an un-folded pair must not widen.
        own = node.n + node.offset
        tightened = own if pending is None else min(pending[0], own)
        _collect(node.input, (tightened, pending[1] if pending else None), caps)
        return
    if isinstance(node, Sort):
        # A sort is where an unordered cap would become unsound — "the first n" of a
        # sorted relation is not the first n of its input — and also the one place a cap
        # can be *rescued*, by pushing the ordering along with it. `topn_fusion` has
        # normally already folded the `Limit` into `Sort.limit` by the time this runs, so
        # that is the shape usually seen here; an un-fused `Limit(Sort(...))` reaches the
        # same answer through `pending`.
        already_ordered = pending is not None and pending[1] is not None
        columns = None if already_ordered else _sort_columns(node)
        counts = [n for n in (node.limit, pending[0] if pending else None) if n is not None]
        if columns is None or not counts:
            _collect(node.input, None, caps)
        else:
            _collect(node.input, (min(counts), columns), caps)
        return
    if isinstance(node, Project):
        # Row-for-row and order-preserving, so the nth projected row is the nth input row.
        _collect(node.input, pending, caps)
        return
    for child in children(node):
        _collect(child, None, caps)
