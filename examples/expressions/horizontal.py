"""Horizontal functions: reducing across columns instead of down rows.

An aggregate like ``sum()`` collapses a column. The ``*_horizontal`` family collapses a
*row* across several columns and leaves the row count alone. That is what you want for a
per-row total, a "did any check fail" flag, or a coalesce-style fallback.

    python examples/expressions/horizontal.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    checks = bt.from_pydict(
        {
            "id": [1, 2, 3],
            "q1": [10, 0, 5],
            "q2": [20, 0, 15],
            "q3": [30, 40, None],
        }
    )

    rolled = checks.with_columns(
        # Numeric reductions across the three quarter columns.
        total=bt.sum_horizontal(col("q1"), col("q2"), col("q3")),
        best=bt.max_horizontal(col("q1"), col("q2"), col("q3")),
        worst=bt.min_horizontal(col("q1"), col("q2"), col("q3")),
        average=bt.mean_horizontal(col("q1"), col("q2"), col("q3")),
        # How many of the listed columns are non-null on this row.
        present=bt.count_horizontal(col("q1"), col("q2"), col("q3")),
        # Boolean reductions.
        all_positive=bt.all_horizontal(col("q1") > 0, col("q2") > 0),
        any_positive=bt.any_horizontal(col("q1") > 0, col("q2") > 0),
        # First non-null wins.
        fallback=bt.coalesce(col("q3"), col("q2"), col("q1")),
        # Concatenate several columns into one string.
        label=bt.concat_str(bt.lit("row-"), col("id").cast("string")),
    )

    result = rolled.to_pydict()
    print(result)

    assert result["total"][0] == 60
    assert result["best"][0] == 30
    assert result["worst"][0] == 10
    assert result["average"][0] == 20.0
    # Row 3 has a null in q3, so only two of the three columns are present.
    assert result["present"] == [3, 3, 2]
    assert result["all_positive"] == [True, False, True]
    assert result["any_positive"] == [True, False, True]
    # Row 3 falls back from the null q3 to q2.
    assert result["fallback"] == [30, 40, 15]
    assert result["label"] == ["row-1", "row-2", "row-3"]

    # The row count is unchanged -- this is a horizontal reduction, not an aggregate.
    assert len(result["total"]) == 3


if __name__ == "__main__":
    main()
