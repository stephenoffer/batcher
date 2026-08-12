"""Helpers shared across `ml` — column checks, indicator casting, and scalar collection.

Small pieces that many modules need and none of them owns. They live here rather than
being pasted into whichever module happened to be written first: `require_columns` had
reached four verbatim copies across `splitting`, `metrics`, and `preprocessors` before
they were collapsed onto this one -- and then **thirty more** across the rest of `ml`,
because six callers importing a helper is not visible from a module that has not imported
it yet, while the four-line inline check is visible in every neighbour. The lesson worth
carrying: a shared helper only wins once every site is actually routed through it, and the
way to keep it that way is a test that fails on a new inline copy
(`tests/unit/test_column_check_is_shared.py`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher.plan.expr_ir.constructors import col

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset
    from batcher.plan.expr_ir import Expr

__all__ = ["indicator", "require_columns", "require_names", "scalar"]


def indicator(name: str) -> Expr:
    """A per-row "did it happen" column as a boolean, whichever way it is encoded.

    A success/correctness flag arrives as either a boolean or a 0/1 integer, and a test has no
    reason to care which. Comparing against a fixed literal does care: `col(x) == 1` fails on a
    boolean column and `col(x) == True` fails on an integer one, both as a raw Arrow
    `RuntimeError` from inside the engine rather than an error at the API edge. Casting is the
    one spelling that accepts both, and it keeps nulls null so a missing observation stays
    missing instead of counting as a failure.

    Args:
        name: The column holding the per-row flag.

    Returns:
        A boolean expression that is true on the rows where the flag is set.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats._shared import indicator
            >>> bt.from_pydict({"hit": [1, 0, 1]}).select(h=indicator("hit")).to_pydict()
            {'h': [True, False, True]}

            >>> bt.from_pydict({"hit": [True, None]}).select(h=indicator("hit")).to_pydict()
            {'h': [True, None]}
    """
    return col(name).cast("boolean")


def require_columns(ds: Dataset, *names: str, hint: str = "Pass an existing column.") -> None:
    """Raise a `ColumnNotFoundError` naming the closest real column for any missing name.

    Args:
        ds: The dataset to check against.
        *names: The column names that must be present.
        hint: The remedy appended to the error, for callers that want a narrower one
            than "pass an existing column" (a projection that needs *numeric* columns,
            say).

    Raises:
        ColumnNotFoundError: On the first name that is not a column of `ds`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats._shared import require_columns
            >>> require_columns(bt.from_pydict({"a": [1]}), "a")
    """
    require_names(ds.columns, *names, hint=hint)


def require_names(
    available: Sequence[str], *names: str, hint: str = "Pass an existing column."
) -> None:
    """`require_columns` for a caller holding a column *list* rather than a `Dataset`.

    The same check, one level down, because not every caller has a `Dataset`: a batch-level
    UDF has ``batch.schema.names``, and a feature-spec validator has the projection it is about
    to build. Both used to inline the check, and one of them then reported the *parameter* it
    wanted rather than the columns that exist — which is the whole thing this error is for.

    Args:
        available: The column names that do exist, in their real order. Order matters: the
            "did you mean" suggestion is drawn from it, and a set would scramble the tie-break.
        *names: The column names that must be present.
        hint: The remedy appended to the error.

    Raises:
        ColumnNotFoundError: On the first name that is not in `available`.

    Examples:
        .. doctest::

            >>> from batcher.ml.stats._shared import require_names
            >>> require_names(["a", "b"], "a", "b")
            >>> require_names(["text"], "txt")
            Traceback (most recent call last):
            batcher._internal.errors.hierarchy.ColumnNotFoundError: Unknown column 'txt'...
    """
    # Membership against a set: the check runs per requested name, and `available` is the
    # relation's full width — a wide feature table turned a handful of name checks into a
    # scan of thousands of columns each. The sequence is kept for the error message, which
    # needs the original order to suggest a close match.
    present = set(available)
    for name in names:
        if name not in present:
            from batcher._internal.errors import ColumnNotFoundError, unknown_message

            raise ColumnNotFoundError(unknown_message("column", name, list(available), hint=hint))


def scalar(ds: Dataset, name: str) -> float:
    """Collect a one-row, one-column aggregate as a Python float.

    An empty input or a null result becomes NaN rather than raising, because a statistic
    over no rows is undefined, not an error.

    Args:
        ds: A dataset reduced to a single row.
        name: The column to read.

    Returns:
        The value as a float, or NaN.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats._shared import scalar
            >>> scalar(bt.from_pydict({"x": [1.0, 3.0]}).agg(m=bt.col("x").mean()), "m")
            2.0
    """
    row = ds.collect()
    if row.num_rows == 0:
        return float("nan")
    value = row.column(name)[0].as_py()
    return float("nan") if value is None else float(value)
