"""Branching inside an expression: when/then/otherwise, and the SQL null helpers.

``bt.when(...).then(...).otherwise(...)`` is the columnar ``if``. Chain ``.when()`` for
more branches; the first matching branch wins, exactly like SQL ``CASE``. Because it is an
expression it runs in Rust, so a five-way bucketing is still one pass.

``.otherwise(...)`` is required, and it needs a real value: there is no null literal, so
an unmatched row must be given an explicit sentinel rather than left null.

    python examples/expressions/conditionals.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    orders = bt.from_pydict(
        {
            "amount": [5, 50, 500, 5000],
            "discount": [0, 0, 10, None],
        }
    )

    labelled = orders.with_columns(
        # A multi-branch bucket. First match wins, `otherwise` is the fallback.
        tier=(
            bt.when(col("amount") < 10)
            .then(bt.lit("micro"))
            .when(col("amount") < 100)
            .then(bt.lit("small"))
            .when(col("amount") < 1000)
            .then(bt.lit("medium"))
            .otherwise(bt.lit("large"))
        ),
        # A two-branch flag.
        free_shipping=bt.when(col("amount") >= 100).then(bt.lit(True)).otherwise(bt.lit(False)),
        # SQL null helpers.
        # `coalesce` -> first non-null; `nullif` -> null when the two sides are equal.
        discount_or_zero=bt.coalesce(col("discount"), bt.lit(0)),
        discount_nonzero=bt.nullif(col("discount"), bt.lit(0)),
        # Row-wise max/min across expressions.
        floor_at_10=bt.greatest(col("amount"), bt.lit(10)),
        cap_at_1000=bt.least(col("amount"), bt.lit(1000)),
    )

    result = labelled.to_pydict()
    print(result)

    assert result["tier"] == ["micro", "small", "medium", "large"]
    assert result["free_shipping"] == [False, False, True, True]
    assert result["discount_or_zero"] == [0, 0, 10, 0]
    # 0 becomes null; 10 stays; an existing null stays null.
    assert result["discount_nonzero"] == [None, None, 10, None]
    assert result["floor_at_10"] == [10, 50, 500, 5000]
    assert result["cap_at_1000"] == [5, 50, 500, 1000]

    # Give unmatched rows an explicit sentinel, then turn it into a null with `nullif`
    # if that is what downstream wants.
    partial = orders.select(
        big=bt.nullif(
            bt.when(col("amount") > 100).then(bt.lit("big")).otherwise(bt.lit("")),
            bt.lit(""),
        )
    ).to_pydict()
    print(partial)
    assert partial["big"] == [None, None, "big", "big"]


if __name__ == "__main__":
    main()
