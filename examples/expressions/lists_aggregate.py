"""Reducing a list column to one value per row.

These are per-row reductions, not group-by aggregates: ``.list.sum()`` sums *within* each
row's list and leaves the row count unchanged. That is the difference between "total per
basket" and "total across baskets".

    python examples/expressions/lists_aggregate.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    readings = bt.from_pydict(
        {
            "sensor": ["s1", "s2"],
            "samples": [[3.0, 1.0, 4.0, 1.0], [10.0, 20.0]],
        }
    )

    stats = readings.with_columns(
        total=col("samples").list.sum(),
        lo=col("samples").list.min(),
        hi=col("samples").list.max(),
        avg=col("samples").list.mean(),
        med=col("samples").list.median(),
        spread=col("samples").list.std(),
        variance=col("samples").list.var(),
        distinct=col("samples").list.n_unique(),
        product=col("samples").list.product(),
        # Index of the smallest / largest element.
        argmin=col("samples").list.arg_min(),
        argmax=col("samples").list.arg_max(),
        # Ordering and de-duplication, still per row.
        sorted_vals=col("samples").list.sort(),
        reversed_vals=col("samples").list.reverse(),
        uniq=col("samples").list.unique(),
        # Running totals and deltas within the row.
        running=col("samples").list.cum_sum(),
        deltas=col("samples").list.diff(),
    )

    result = stats.to_pydict()
    print(result)

    assert result["total"] == [9.0, 30.0]
    assert result["lo"] == [1.0, 10.0]
    assert result["hi"] == [4.0, 20.0]
    assert result["avg"] == [2.25, 15.0]
    assert result["distinct"] == [3, 2]
    assert result["product"] == [12.0, 200.0]
    assert result["argmin"][0] == 1
    assert result["argmax"][0] == 2
    assert result["sorted_vals"][0] == [1.0, 1.0, 3.0, 4.0]
    assert result["reversed_vals"][0] == [1.0, 4.0, 1.0, 3.0]
    assert result["running"][0] == [3.0, 4.0, 8.0, 9.0]
    # The row count is unchanged: this is a per-row reduction.
    assert len(result["total"]) == 2


if __name__ == "__main__":
    main()
