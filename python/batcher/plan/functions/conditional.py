"""Conditional / null-handling free functions (`iff`, `nanvl`, `cut`).

Thin sugar over `when().then().otherwise()` for the common cases users reach for from
DuckDB/Spark: the two-branch `iff`, the NaN-replacing `nanvl`, and the multi-branch `cut`
that bins a numeric column by explicit edges. No new IR — each lowers to a `when` chain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.constructors import col, lit, when
from batcher.plan.expr_ir.core import Expr, IntoExpr, _wrap

if TYPE_CHECKING:
    from collections.abc import Sequence


def iff(condition: Expr, if_true: IntoExpr, if_false: IntoExpr) -> Expr:
    """``if_true`` where `condition` is true, else ``if_false`` (DuckDB ``IF``/``IFF``).

    The two-branch shorthand for ``when(condition).then(if_true).otherwise(if_false)``.

    Args:
        condition: The boolean predicate selecting the branch per row.
        if_true: The value where ``condition`` is true.
        if_false: The value where ``condition`` is false or null.

    Returns:
        An expression yielding ``if_true`` or ``if_false`` per row.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [-1, 2]})
            >>> ds.select(s=bt.iff(bt.col("x") > 0, bt.lit("pos"), bt.lit("neg"))).to_pydict()
            {'s': ['neg', 'pos']}
    """
    return when(condition).then(_wrap(if_true)).otherwise(_wrap(if_false))


def nanvl(value: IntoExpr, fallback: IntoExpr) -> Expr:
    """`value` unless it is NaN, in which case `fallback` (Spark ``nanvl``).

    Distinct from `coalesce` — this replaces IEEE NaN, not NULL. A NULL `value`
    passes through unchanged (NULL is not NaN).

    Args:
        value: The value to return unless it is NaN.
        fallback: The replacement used where ``value`` is NaN.

    Returns:
        An expression yielding ``value``, or ``fallback`` where ``value`` is NaN.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, float("nan")]})
            >>> ds.select(r=bt.nanvl(bt.col("x"), bt.lit(0.0))).to_pydict()
            {'r': [1.0, 0.0]}
    """
    v = _wrap(value)
    return when(v.is_nan()).then(_wrap(fallback)).otherwise(v)


def cut(
    value: IntoExpr,
    breaks: Sequence[float],
    *,
    labels: Sequence[object] | None = None,
    right: bool = True,
) -> Expr:
    """Bin a numeric column into buckets defined by explicit edges.

    With ``n`` sorted break points the column is split into ``n + 1`` buckets: everything at or
    below the first break, each interval between consecutive breaks, and everything above the
    last. By default a value equal to a break falls into the lower bucket (``right=True``,
    left-open intervals ``(a, b]``), matching ``pandas.cut`` and the usual "up to and including"
    reading of a threshold. Set ``right=False`` for right-open intervals ``[a, b)``.

    The result is the integer bin index by default, or the matching entry from `labels` when
    given. It lowers to a `when`/`then` chain, so it is a pure per-row expression with no `fit`
    and no aggregate. Reach for `cut` when the edges are known up front and for
    `KBinsDiscretizer` when they must be learned from the data.

    Args:
        value: The numeric column (or expression) to bin.
        breaks: The sorted, strictly increasing interior edge values. ``n`` breaks yield
            ``n + 1`` buckets.
        labels: One label per bucket (so ``len(breaks) + 1`` of them) to return instead of the
            integer index. Omit to return the 0-based bin index.
        right: If true (default), intervals are left-open ``(a, b]`` and a value equal to a
            break goes to the lower bucket. If false, intervals are right-open ``[a, b)``.

    Returns:
        An expression giving each row's bucket, as an integer index or a `labels` entry.

    Raises:
        PlanError: If `breaks` is empty, or `labels` is given with the wrong length.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"age": [5, 18, 40, 70]})
            >>> ds.with_columns(band=bt.cut("age", [12, 19, 65])).to_pydict()["band"]
            [0, 1, 2, 3]

            >>> labeled = bt.cut("age", [12, 19, 65], labels=["child", "teen", "adult", "senior"])
            >>> ds.with_columns(band=labeled).to_pydict()["band"]
            ['child', 'teen', 'adult', 'senior']
    """
    if len(breaks) == 0:
        raise PlanError("cut needs at least one break point.")
    if labels is not None and len(labels) != len(breaks) + 1:
        raise PlanError(
            f"cut got {len(breaks)} breaks (so {len(breaks) + 1} buckets) but "
            f"{len(labels)} labels; pass one label per bucket."
        )
    column = col(value) if isinstance(value, str) else _wrap(value)
    outputs: list[object] = list(labels) if labels is not None else list(range(len(breaks) + 1))
    # Build the when/then chain from the last edge inward so the earliest (lowest) edge a value
    # falls under wins. A left-open interval (a, b] means "value <= b"; a right-open one [a, b)
    # means "value < b". The final bucket is the else branch.
    chain: Expr = _wrap(outputs[-1])
    for index in range(len(breaks) - 1, -1, -1):
        condition = column <= lit(breaks[index]) if right else column < lit(breaks[index])
        chain = when(condition).then(_wrap(outputs[index])).otherwise(chain)
    return chain
