"""The aggregate vocabulary: counts, positions, quantiles, and approximations.

Exact aggregates read every row. The ``approx_*`` family reads sketches instead, trading a
bounded error for a large constant-factor speedup and, more importantly, bounded memory on
a high-cardinality column.

    python examples/expressions/aggregates.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    ds = bt.from_pydict(
        {
            "grp": ["a", "a", "a", "b", "b"],
            "v": [3, 1, 4, 1, 5],
            "w": [10, 20, 30, 40, 50],
            "flag": [True, False, True, True, False],
        }
    )

    agg = ds.select(
        n=bt.count(),
        non_null=col("v").count(),
        distinct=col("v").n_unique(),
        total=col("v").sum(),
        smallest=col("v").min(),
        largest=col("v").max(),
        average=col("v").mean(),
        middle=col("v").median(),
        spread=col("v").std(),
        variance=col("v").var(),
        p90=col("v").quantile(0.9),
        # First/last by another column's order.
        first_v=bt.first("v", order_by="w"),
        last_v=bt.last("v", order_by="w"),
        # The value of one column at the row where another is extreme.
        w_at_max_v=bt.arg_max("w", col("v")),
        w_at_min_v=bt.arg_min("w", col("v")),
        # Boolean reductions.
        any_flag=col("flag").any(),
        all_flag=col("flag").all(),
        # Bitwise reductions.
        bits_or=bt.bit_or("v"),
        bits_and=bt.bit_and("v"),
        bits_xor=bt.bit_xor("v"),
    ).to_pydict()

    print(agg)

    assert agg["n"] == [5]
    assert agg["distinct"] == [4]  # 3, 1, 4, 5
    assert agg["total"] == [14]
    assert agg["smallest"] == [1] and agg["largest"] == [5]
    assert agg["middle"] == [3]
    assert agg["first_v"] == [3] and agg["last_v"] == [5]
    # `v` peaks at 5, where `w` is 50.
    assert agg["w_at_max_v"] == [50]
    assert agg["any_flag"] == [True]
    assert agg["all_flag"] == [False]

    # Approximate aggregates: sketch-backed, bounded memory.
    approx = ds.select(
        distinct=col("v").approx_n_unique(),
        median=col("v").approx_median(),
        p90=col("v").approx_quantile(0.9),
    ).to_pydict()
    print("approx:", approx)
    assert approx["distinct"][0] >= 3

    # Every one of these also works per group, in a single pass.
    grouped = (
        ds.group_by("grp")
        .agg(total=col("v").sum(), spread=col("v").max() - col("v").min(), n=bt.count())
        .sort("grp")
        .to_pydict()
    )
    print(grouped)
    assert grouped["total"] == [8, 6]
    assert grouped["spread"] == [3, 4]


if __name__ == "__main__":
    main()
