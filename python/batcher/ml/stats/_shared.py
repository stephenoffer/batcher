"""Helpers shared by the statistical modules — column checks and scalar collection.

Two one-liners that every module in this package needs and none of them owns. They live
here rather than being pasted three times, and rather than living in whichever module
happened to be written first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["require_columns", "scalar"]


def require_columns(ds: Dataset, *names: str) -> None:
    """Raise a `ColumnNotFoundError` naming the closest real column for any missing name.

    Args:
        ds: The dataset to check against.
        *names: The column names that must be present.

    Raises:
        ColumnNotFoundError: On the first name that is not a column of `ds`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats._shared import require_columns
            >>> require_columns(bt.from_pydict({"a": [1]}), "a")
    """
    available = ds.columns
    for name in names:
        if name not in available:
            from batcher._internal.errors import ColumnNotFoundError, unknown_message

            raise ColumnNotFoundError(
                unknown_message("column", name, available, hint="Pass an existing column.")
            )


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
