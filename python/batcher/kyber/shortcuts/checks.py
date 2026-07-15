"""Predicate shortcuts — questions about a column's *values*, answered from its bounds.

This is where a pair of numbers stops being a statistic and starts being an answer. A min
and a max are values that **occur in the column**, and that turns a whole class of predicate
questions into arithmetic:

  - "is every value positive?" is decided by the minimum, alone;
  - "is any value above the threshold?" is decided by the maximum, alone;
  - "does the column contain 42?" is *refuted* whenever 42 lies outside `[min, max]`, or a
    membership bloom rejects it — and refutation is the answer that saves the scan, because
    it is the one a filter can act on.

Every function is over the column's **non-null** values, the way SQL comparisons are: a null
neither satisfies nor violates. A column with no non-null value at all satisfies every
`all_*` check vacuously and no `any_*` check — stated here once so a hundred call sites do
not each have to decide.

The float gate lives in `bounds.orderable`: a column that may hold a NaN answers nothing
here, because a bound that dropped a NaN describes a column that is not the one on disk.
"""

from __future__ import annotations

from typing import Any

from batcher.kyber.shortcuts.bounds import orderable
from batcher.kyber.shortcuts.facts import Facts

__all__ = [
    "all_between",
    "all_greater_equal",
    "all_greater_than",
    "all_less_equal",
    "all_less_than",
    "all_negative",
    "all_non_negative",
    "all_non_positive",
    "all_positive",
    "all_zero",
    "any_greater_equal",
    "any_greater_than",
    "any_less_equal",
    "any_less_than",
    "contains",
    "may_contain",
]


def _vacuous(facts: Facts, column: str) -> bool | None:
    """Whether `column` provably has no non-null value (so every `all_*` check holds).

    True for an empty relation and for an all-null column; False when there is provably at
    least one non-null value; None when the counts are not exact and nothing can be said.
    """
    nulls = facts.col(column).null_count
    if facts.rows is None or nulls is None:
        return None
    return facts.rows - nulls == 0


def _compare(facts: Facts, column: str, bound: str, op: Any, value: Any) -> bool | None:
    """Decide one bound-vs-literal comparison, honouring the vacuous and float rules.

    `bound` is ``"min"`` or ``"max"``: the single bound that decides this predicate. An
    incomparable literal (a string against a numeric column) is not an error here — it is a
    question metadata cannot answer, so it returns None and the engine answers it.
    """
    col = orderable(facts, column)
    if col is None:
        return None
    edge = col.min if bound == "min" else col.max
    if edge is None:
        return None
    try:
        return bool(op(edge, value))
    except TypeError:  # incomparable types — let the engine decide (or raise) on real values
        return None


def _all(facts: Facts, column: str, bound: str, op: Any, value: Any) -> bool | None:
    """An `all_*` check: vacuously true with no non-null value, else the bound decides."""
    if _vacuous(facts, column):
        return True
    return _compare(facts, column, bound, op, value)


def _any(facts: Facts, column: str, bound: str, op: Any, value: Any) -> bool | None:
    """An `any_*` check: false with no non-null value, else the bound decides."""
    if _vacuous(facts, column):
        return False
    return _compare(facts, column, bound, op, value)


def _gt(a: Any, b: Any) -> bool:
    return a > b


def _ge(a: Any, b: Any) -> bool:
    return a >= b


def _lt(a: Any, b: Any) -> bool:
    return a < b


def _le(a: Any, b: Any) -> bool:
    return a <= b


def all_greater_than(facts: Facts, column: str, value: Any) -> bool | None:
    """Whether every non-null value exceeds `value`, or None when not provable.

    Decided by the minimum alone — it is the smallest value that occurs, so if it clears the
    threshold, everything does.
    """
    return _all(facts, column, "min", _gt, value)


def all_greater_equal(facts: Facts, column: str, value: Any) -> bool | None:
    """Whether every non-null value is at least `value`, or None when not provable."""
    return _all(facts, column, "min", _ge, value)


def all_less_than(facts: Facts, column: str, value: Any) -> bool | None:
    """Whether every non-null value is below `value`, or None when not provable.

    The mirror of `all_greater_than`, decided by the maximum alone.
    """
    return _all(facts, column, "max", _lt, value)


def all_less_equal(facts: Facts, column: str, value: Any) -> bool | None:
    """Whether every non-null value is at most `value`, or None when not provable."""
    return _all(facts, column, "max", _le, value)


def any_greater_than(facts: Facts, column: str, value: Any) -> bool | None:
    """Whether some non-null value exceeds `value`, or None when not provable.

    Decided by the maximum alone, in *both* directions: a max above the threshold proves a
    match exists, and a max at or below it proves none does. That second half is what lets a
    `WHERE x > 1000` be answered as "no rows" without opening a file.
    """
    return _any(facts, column, "max", _gt, value)


def any_greater_equal(facts: Facts, column: str, value: Any) -> bool | None:
    """Whether some non-null value is at least `value`, or None when not provable."""
    return _any(facts, column, "max", _ge, value)


def any_less_than(facts: Facts, column: str, value: Any) -> bool | None:
    """Whether some non-null value is below `value`, or None when not provable."""
    return _any(facts, column, "min", _lt, value)


def any_less_equal(facts: Facts, column: str, value: Any) -> bool | None:
    """Whether some non-null value is at most `value`, or None when not provable."""
    return _any(facts, column, "min", _le, value)


def all_between(facts: Facts, column: str, low: Any, high: Any) -> bool | None:
    """Whether every non-null value lies in ``[low, high]`` (inclusive), or None if not provable.

    Both bounds must clear their side: it is the conjunction of `all_greater_equal(low)` and
    `all_less_equal(high)`, and it needs both a min and a max to decide. The range-check a
    data-quality gate runs, answered from the footer instead of from every row.
    """
    lower = all_greater_equal(facts, column, low)
    upper = all_less_equal(facts, column, high)
    if lower is None or upper is None:
        return None
    return lower and upper


def all_positive(facts: Facts, column: str) -> bool | None:
    """Whether every non-null value is strictly greater than zero, or None if not provable."""
    return all_greater_than(facts, column, 0)


def all_non_negative(facts: Facts, column: str) -> bool | None:
    """Whether every non-null value is zero or greater, or None when not provable."""
    return all_greater_equal(facts, column, 0)


def all_negative(facts: Facts, column: str) -> bool | None:
    """Whether every non-null value is strictly less than zero, or None if not provable."""
    return all_less_than(facts, column, 0)


def all_non_positive(facts: Facts, column: str) -> bool | None:
    """Whether every non-null value is zero or less, or None when not provable."""
    return all_less_equal(facts, column, 0)


def all_zero(facts: Facts, column: str) -> bool | None:
    """Whether every non-null value is exactly zero, or None when not provable."""
    return all_between(facts, column, 0, 0)


def contains(facts: Facts, column: str, value: Any) -> bool | None:
    """Whether `column` holds `value` somewhere, or None when metadata cannot decide.

    Metadata is *asymmetric* here, and that asymmetry is the point:

    * **Absence is provable.** A value outside ``[min, max]``, or one a membership bloom
      rejects, is not in the column — and cannot be in any subset of it, which is why the
      bloom survives a filter that invalidates every other statistic. This is the answer that
      skips the scan.
    * **Presence is (almost) not.** Bounds say a value *could* be there, not that it is. The
      one exception is a constant column, where `min == max == value` proves it.

    Anything else returns None, and the engine looks.
    """
    col = orderable(facts, column)
    if col is None:
        return None
    if _refuted_by_bounds(col, value) or _refuted_by_bloom(col, value):
        return False
    if col.has_bounds and col.min == col.max and col.min == value:
        return True  # a constant column: the one value it holds is this one
    return None


def may_contain(facts: Facts, column: str, value: Any) -> bool:
    """Whether `column` *might* hold `value` — never executes, and never says "no" wrongly.

    The one-sided form of `contains`, for a caller that wants a cheap filter rather than an
    answer: ``False`` is a proof of absence (bounds or bloom refute it), ``True`` means
    "metadata cannot rule it out", which is not the same as "it is there". Skipping a file,
    a partition, or a whole query on a ``False`` is always sound.
    """
    col = orderable(facts, column)
    if col is None:
        return True
    return not (_refuted_by_bounds(col, value) or _refuted_by_bloom(col, value))


def _refuted_by_bounds(col: Any, value: Any) -> bool:
    """Whether `value` provably lies outside the column's `[min, max]` range."""
    if not col.has_bounds:
        return False
    try:
        return bool(value < col.min or value > col.max)
    except TypeError:  # incomparable — proves nothing
        return False


def _refuted_by_bloom(col: Any, value: Any) -> bool:
    """Whether the column's membership bloom proves `value` absent.

    Sound at any provenance: a bloom is built from the values that *are* present, and
    removing rows never adds one — so "absent from the bloom" stays a proof of absence over
    any subset of the relation. A false positive is possible (the bloom may say "maybe" for a
    value that is not there); a false negative is not, which is the only direction we rely on.
    """
    if col.bloom is None:
        return False
    from batcher.plan.bloom_index import BloomIndex

    index = BloomIndex.from_bytes(col.bloom)
    return index is not None and not index.contains(value)
