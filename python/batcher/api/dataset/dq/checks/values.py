"""Value constraints: presence, bounds, membership, sign, and numeric well-formedness.

Every constraint here except `not_null` treats NULL as valid, so an optional column does
not fail every check written against it — the dbt/Great-Expectations convention. Forbid
nulls explicitly with `not_null`, which composes with any of them.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import reduce
from typing import Any

from batcher._internal.errors import PlanError
from batcher.api.dataset.dq.constraints import RowConstraint
from batcher.plan.expr_ir import Col

__all__ = [
    "accepted_values",
    "in_range",
    "is_finite",
    "not_null",
    "positive",
    "rejected_values",
]


def not_null(cols: tuple[str, ...]) -> RowConstraint:
    """Every column in `cols` must be non-null.

    Args:
        cols: The columns that must not contain a null.

    Returns:
        The row constraint.
    """
    if not cols:
        raise PlanError("not_null() requires at least one column, e.g. not_null('id')")
    valid = reduce(lambda a, b: a & b, (Col(c).is_not_null() for c in cols))
    return RowConstraint(f"not_null({', '.join(cols)})", valid)


def in_range(column: str, low: Any, high: Any, closed: str = "both") -> RowConstraint:
    """`column` must lie between `low` and `high` (NULL passes).

    Args:
        column: The column to bound.
        low: Lower bound.
        high: Upper bound.
        closed: Which ends are inclusive — `"both"`, `"left"`, `"right"`, or `"none"`.

    Returns:
        The row constraint.
    """
    if low > high:
        raise PlanError(
            f"in_range({column!r}): low ({low!r}) > high ({high!r}) — swap the arguments?"
        )
    if closed not in ("both", "left", "right", "none"):
        raise PlanError(
            f"in_range({column!r}, closed={closed!r}): use 'both', 'left', 'right', or 'none'."
        )
    c = Col(column)
    suffix = "" if closed == "both" else f", closed={closed}"
    return RowConstraint(
        f"in_range({column}, {low}, {high}{suffix})",
        c.is_null() | c.between(low, high, closed=closed),
    )


def accepted_values(column: str, allowed: Iterable[Any]) -> RowConstraint:
    """`column` must be one of `allowed` (NULL passes).

    Args:
        column: The column to test.
        allowed: The permitted set of values.

    Returns:
        The row constraint.
    """
    members = list(allowed)
    if not members:
        # An empty allow-list makes every non-null row a violation, so `.drop()`
        # silently empties the dataset. That is never the intent — it is a config
        # value that arrived empty — and every sibling that takes a set of values
        # (`classify(labels=)`, `random_split`) already refuses one.
        raise PlanError(
            f"accepted_values({column!r}): values must be non-empty; an empty "
            "allow-list rejects every row."
        )
    c = Col(column)
    return RowConstraint(f"accepted_values({column})", c.is_null() | c.is_in(members))


def rejected_values(column: str, forbidden: Iterable[Any]) -> RowConstraint:
    """`column` must be none of `forbidden` (NULL passes).

    The complement of `accepted_values`, and the right shape for a deny-list: the sentinel
    values an upstream system writes for "unknown" (`"N/A"`, `-1`, `"null"`) are a closed
    set, while the values you actually accept are not.

    Args:
        column: The column to test.
        forbidden: The values that must not appear.

    Returns:
        The row constraint.
    """
    members = list(forbidden)
    if not members:
        raise PlanError(
            f"rejected_values({column!r}): values must be non-empty; an empty "
            "deny-list rejects nothing, so the constraint would do nothing."
        )
    c = Col(column)
    return RowConstraint(f"rejected_values({column})", c.is_null() | ~c.is_in(members))


def positive(column: str, *, strict: bool = True) -> RowConstraint:
    """`column` must be greater than zero — or at least zero when `strict=False`.

    Args:
        column: The numeric column to test.
        strict: Whether zero is a violation.

    Returns:
        The row constraint.
    """
    c = Col(column)
    test = (c > 0) if strict else (c >= 0)
    label = "positive" if strict else "non_negative"
    return RowConstraint(f"{label}({column})", c.is_null() | test)


def is_finite(column: str) -> RowConstraint:
    """`column` must be a finite number — no NaN and no infinity (NULL passes).

    The check a float column needs and `in_range` cannot give it: NaN compares false
    against every bound, so ``in_range(x, 0, 100)`` already rejects a NaN, but ``x > 0``
    written as a `check` does too, quietly, and neither says NaN is why. Infinity is the
    opposite failure — it passes a lower bound and destroys every downstream mean.

    Args:
        column: The float column to test.

    Returns:
        The row constraint.
    """
    c = Col(column)
    return RowConstraint(f"is_finite({column})", c.is_null() | c.is_finite())
