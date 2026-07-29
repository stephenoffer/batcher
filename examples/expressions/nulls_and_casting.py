"""Nulls and type casting: the two places a pipeline quietly changes its answer.

Null is not zero and not empty string, and every aggregate skips it. Casting is where a
schema mismatch between two sources gets resolved, and where an unparseable value becomes
a null rather than an error.

    python examples/expressions/nulls_and_casting.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    data = bt.from_pydict(
        {
            "amount": [10.0, None, 30.0, None],
            "code": ["1", "2", "oops", "4"],
            "flag": [1, 0, 1, 0],
        }
    )

    handled = data.with_columns(
        missing=col("amount").is_null(),
        present=col("amount").is_not_null(),
        filled=col("amount").fill_null(0.0),
        # `coalesce` generalizes fill_null to several fallbacks.
        fallback=bt.coalesce(col("amount"), col("flag").cast("double"), bt.lit(-1.0)),
    ).to_pydict()

    print(handled)
    assert handled["missing"] == [False, True, False, True]
    assert handled["present"] == [True, False, True, False]
    assert handled["filled"] == [10.0, 0.0, 30.0, 0.0]
    # Row 2's amount is null, so it falls through to `flag` (0 -> 0.0).
    assert handled["fallback"] == [10.0, 0.0, 30.0, 0.0]

    # Aggregates skip nulls rather than propagating them.
    agg = data.select(
        total=col("amount").sum(),
        mean=col("amount").mean(),
        n=bt.count(),
        non_null=col("amount").count(),
    ).to_pydict()
    print(agg)
    assert agg["total"] == [40.0]
    assert agg["mean"] == [20.0]  # 40 / 2, not 40 / 4
    assert agg["n"] == [4]
    assert agg["non_null"] == [2]

    # Casting comes in two flavours. `cast` is strict: a value it cannot parse raises,
    # so a malformed row stops the job rather than becoming a silent null.
    try:
        data.select(as_int=col("code").cast("int64")).to_pydict()
    except Exception as exc:
        print("strict cast raised:", type(exc).__name__)
    else:
        raise AssertionError("expected the strict cast to raise")

    # `try_cast` is the lenient one: unparseable values become null.
    lenient = data.select(
        as_int=col("code").try_cast("int64"),
        as_text=col("flag").cast("string"),
        as_bool=col("flag").cast("boolean"),
    ).to_pydict()
    print(lenient)
    assert lenient["as_int"] == [1, 2, None, 4]
    assert lenient["as_text"] == ["1", "0", "1", "0"]
    assert lenient["as_bool"] == [True, False, True, False]

    # With `try_cast` the failure is silent, so count the nulls it introduced.
    before = data.select(n=col("code").count()).to_pydict()["n"][0]
    after = data.select(n=col("code").try_cast("int64").count()).to_pydict()["n"][0]
    print("parsed", after, "of", before)
    assert after < before  # one value failed to parse

    # Float edge cases.
    floats = bt.from_pydict({"v": [1.0, float("nan"), float("inf"), -1.0]})
    edges = floats.select(nan=col("v").is_nan(), finite=col("v").is_finite()).to_pydict()
    print(edges)
    assert edges["nan"] == [False, True, False, False]
    assert edges["finite"] == [True, False, False, True]


if __name__ == "__main__":
    main()
