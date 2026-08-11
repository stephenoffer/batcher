"""Profiling a table before you write any checks.

Writing a contract without profiling first is guessing. These are the numbers that tell
you what to assert: null rate, cardinality, range, and the distribution's shape.

    python examples/quality/profiling_columns.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders")

    # The per-column statistical summary.
    summary = orders.select("o_totalprice", "o_custkey").describe().to_pydict()
    print(summary["statistic"])
    assert "mean" in summary["statistic"]

    # Null counts, column by column.
    nulls = orders.null_count().to_pydict()
    print("nulls:", nulls)
    assert all(value == 0 for column in nulls.values() for value in column)

    # Cardinality tells you whether a column is a key, a category, or free text.
    for column in ("o_orderkey", "o_orderstatus", "o_clerk"):
        distinct = orders.n_unique(column)
        ratio = distinct / orders.count()
        kind = "key" if ratio > 0.99 else ("category" if distinct < 50 else "high-cardinality")
        print(f"{column:<16} {distinct:>7} distinct ({ratio:.4f}) -> {kind}")

    assert orders.n_unique("o_orderkey") == orders.count()
    assert orders.n_unique("o_orderstatus") < 10

    # Range and shape, which is what an `in_range` check needs.
    shape = orders.agg(
        low=col("o_totalprice").min(),
        high=col("o_totalprice").max(),
        median=bt.median(col("o_totalprice")),
        skew=bt.skewness(col("o_totalprice")),
    ).to_pydict()
    print({name: round(value[0], 2) for name, value in shape.items()})
    assert shape["low"][0] < shape["median"][0] < shape["high"][0]

    # `glimpse` and `info` are the interactive forms of the same thing.
    orders.select("o_orderkey", "o_orderstatus", "o_totalprice").glimpse()


if __name__ == "__main__":
    main()
