"""Profiling one column: bounds, uniqueness, nulls, and constancy.

This is the accessor to reach for before writing a data-quality rule. Rather than guessing
a threshold, ask the column what it actually contains, then encode the answer as a check.

    python examples/dataset/meta_columns.py
"""

from __future__ import annotations

import batcher as bt


def main() -> None:
    ds = bt.from_pydict(
        {
            "id": [1, 2, 3, 4],
            "score": [10.0, 20.0, 30.0, None],
            "grade": ["a", "b", "a", "b"],
            "version": [7, 7, 7, 7],
            "flag": [0, 1, 1, 0],
        }
    )

    score = ds.meta.col("score")

    # Range and centre.
    print("bounds:", score.bounds())
    assert score.bounds() == (10.0, 30.0)
    assert score.range() == 20.0
    assert score.midpoint() == 20.0
    assert score.abs_max() == 30.0
    assert score.sum() == 60.0
    assert score.mean() == 20.0

    # Nulls.
    assert score.null_fraction() == 0.25
    assert not score.no_nulls()
    assert ds.meta.col("id").no_nulls()

    # Uniqueness -- the question behind "can I join on this?".
    assert ds.meta.col("id").is_unique()
    assert ds.meta.col("id").is_key()
    assert ds.meta.col("grade").has_duplicates()
    assert ds.meta.col("grade").n_unique() == 2
    assert ds.meta.col("grade").duplicate_count() == 2

    # A column that never varies carries no information.
    version = ds.meta.col("version")
    assert version.is_constant()
    assert version.constant_value() == 7

    # Cardinality and shape hints that drive an encoder choice.
    assert ds.meta.col("grade").is_low_cardinality(max_distinct=10)
    assert ds.meta.col("flag").is_binary_valued()

    # Everything at once.
    summary = score.summary()
    print("summary:", summary)
    assert isinstance(summary, dict)

    # Dataset-level null accounting.
    nulls = ds.meta.nulls
    print("null counts:", nulls.counts())
    assert nulls.any()
    assert not nulls.is_complete()
    assert nulls.total() == 1
    assert nulls.columns_with_nulls() == ["score"]
    assert "id" in nulls.complete_columns()
    assert nulls.fractions()["score"] == 0.25

    # A composite key check, for a table with no single unique column.
    assert not ds.meta.is_key("grade")
    assert ds.meta.is_key(["id", "grade"])


if __name__ == "__main__":
    main()
