"""Join shortcuts — the questions two relations' bounds answer about their join.

The one that pays for itself: if the left key's range is `[1, 10]` and the right key's is
`[900, 999]`, no value can match, and the inner join is **empty** — provably, from four
numbers, before either side has been read. That is not a micro-optimisation on a small query;
it is the difference between a shuffle and no shuffle on a large one, and it is exactly the
case a partitioned fact table hits when a dimension filter selects a range the fact table does
not cover.

The proof relies on min and max being *values that occur* (a `Provenance.EXACT` bound), which
is what `Facts` guarantees. Anything weaker returns None and the join runs.
"""

from __future__ import annotations

from typing import Any

from batcher.kyber.shortcuts.bounds import orderable
from batcher.kyber.shortcuts.facts import Facts

__all__ = ["estimated_join_rows", "join_is_empty", "key_overlap"]


def join_is_empty(left: Facts, right: Facts, left_key: str, right_key: str) -> bool | None:
    """Whether an equi-join on these keys provably yields no row, or None if not provable.

    True when the two key ranges are **disjoint**: every left value is below every right value
    (or above every right one), so no pair can be equal. Only ``True`` is ever proved —
    overlapping ranges do not imply a match exists (the two columns may interleave without
    sharing a value), so an overlap returns None and the join decides.

    Either side being *empty* also proves it: an inner join with no rows on one side has no
    rows at all.
    """
    if left.rows == 0 or right.rows == 0:
        return True  # an inner join against nothing is nothing
    lhs = orderable(left, left_key)
    rhs = orderable(right, right_key)
    if lhs is None or rhs is None or not lhs.has_bounds or not rhs.has_bounds:
        return None
    try:
        disjoint = lhs.max < rhs.min or rhs.max < lhs.min
    except TypeError:  # incomparable key types — the join will raise or coerce; let it
        return None
    # Disjoint proves emptiness. An **overlap proves nothing** — and returning `False` here
    # would be a wrong answer, not a weak one: two key columns can share a range and still
    # share no value (left `{1, 5}` against right `{3}` both live in `[1, 5]`, and their join
    # is empty). Only `True` is a proof; anything else runs the join.
    return True if disjoint else None


def key_overlap(left: Facts, right: Facts, left_key: str, right_key: str) -> tuple[Any, Any] | None:
    """The range both keys share, as `(low, high)`, or None when it cannot be computed.

    The intersection of the two bounds — the only window in which a matching value can lie. A
    `low` above `high` means the ranges are disjoint (and `join_is_empty` says so); an equal
    pair means at most one value can match. Useful for choosing a range partition that both
    sides can be pruned to.
    """
    lhs = orderable(left, left_key)
    rhs = orderable(right, right_key)
    if lhs is None or rhs is None or not lhs.has_bounds or not rhs.has_bounds:
        return None
    try:
        return (max(lhs.min, rhs.min), min(lhs.max, rhs.max))
    except TypeError:
        return None


def estimated_join_rows(left: Facts, right: Facts, left_key: str, right_key: str) -> float:
    """The estimated size of the equi-join result — the classic containment estimate.

    ``|L| * |R| / max(ndv_L, ndv_R)``: each distinct key value on the smaller-cardinality side
    is assumed to match every row carrying it on the other. Explicitly approximate (it reads
    sketched distinct counts and estimated row counts), and it is the number the optimizer
    itself orders joins by — so exposing it is exposing the plan's own reasoning, not a
    separate guess.

    Falls back to the cross-product size when neither key has a known distinct count, which is
    the honest worst case rather than an invented one.

    A join `join_is_empty` **proves** empty estimates zero rather than running the containment
    formula. Without that, the same two `Facts` answered "this join is provably empty" and
    "this join produces two rows" at the same time -- and since this is the number the
    optimizer orders joins by, the one join it could have skipped entirely was costed as
    though it produced output.
    """
    if join_is_empty(left, right, left_key, right_key):
        return 0.0
    rows = left.estimated_rows * right.estimated_rows
    ndvs = [
        ndv
        for ndv in (left.col(left_key).approx_ndv, right.col(right_key).approx_ndv)
        if ndv is not None and ndv > 0
    ]
    if not ndvs:
        return rows
    return rows / max(ndvs)
