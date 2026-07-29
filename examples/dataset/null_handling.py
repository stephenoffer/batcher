"""Dataset-level null handling: dropping, filling, and counting missing values.

Null propagation is where a pipeline changes its answer without telling you. Decide per
column whether a missing value means "unknown" (leave it), "zero" (fill it), or "this row
is unusable" (drop it) -- and never let the default decide for you.

    python examples/dataset/null_handling.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    ds = bt.from_pydict(
        {
            "id": [1, 2, 3, 4],
            "amount": [10.0, None, 30.0, None],
            "note": ["a", "b", None, None],
        }
    )

    # Counting, before deciding.
    assert ds.n_null("amount") == 2
    assert ds.has_nulls("note")
    print("null counts:", ds.null_count().to_pydict())
    print("null fractions:", ds.meta.nulls.fractions())

    # Drop rows with any null.
    complete = ds.drop_nulls().to_pydict()
    print("complete rows:", complete["id"])
    assert complete["id"] == [1]

    # Drop only where a specific column is null -- usually what was meant.
    has_amount = ds.drop_nulls(["amount"]).to_pydict()
    assert has_amount["id"] == [1, 3]

    # Fill everything with one value.
    filled = ds.fill_null(0).to_pydict()
    assert None not in filled["amount"]

    # Per-column fills, which is almost always the right granularity: zero for a
    # measurement, a sentinel for a label.
    per_column = ds.with_columns(
        amount=col("amount").fill_null(0.0),
        note=col("note").fill_null("unknown"),
    ).to_pydict()
    print(per_column)
    assert per_column["amount"] == [10.0, 0.0, 30.0, 0.0]
    assert per_column["note"] == ["a", "b", "unknown", "unknown"]

    # The pandas spellings do the same thing.
    assert ds.dropna(["amount"]).count() == 2
    assert None not in ds.fillna(0).to_pydict()["amount"]

    # Null-aware predicates, as boolean columns.
    flags = ds.select(
        missing=col("amount").is_null(), present=col("amount").is_not_null()
    ).to_pydict()
    assert flags["missing"] == [False, True, False, True]

    # `isna`/`notna` give the same thing at the Dataset level.
    na = ds.isna().to_pydict()
    assert isinstance(na, dict)

    # Why it matters: the mean over four rows with two nulls is over *two* values, and
    # filling with zero gives a different -- and usually wrong -- answer.
    skipped = ds.select(m=col("amount").mean()).to_pydict()["m"][0]
    zero_filled = ds.fill_null(0).select(m=col("amount").mean()).to_pydict()["m"][0]
    print("mean skipping nulls:", skipped, "| mean after fill_null(0):", zero_filled)
    assert skipped == 20.0
    assert zero_filled == 10.0
    assert skipped != zero_filled


if __name__ == "__main__":
    main()
