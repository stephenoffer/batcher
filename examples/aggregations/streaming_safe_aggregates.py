"""Which aggregates can be maintained incrementally, and which cannot.

Sum, count, min and max fold into a fixed-size state. Median and exact distinct do not — they
need the values, so their state grows with the data. Knowing which is which is what decides
whether a metric can run on a stream at all.

    python examples/aggregations/streaming_safe_aggregates.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem").select("l_shipmode", "l_quantity", "l_partkey")
    shards = [lineitem.slice(start, 50_000) for start in range(0, 200_000, 50_000)]

    # Bounded state: these fold.
    partials = [
        shard.group_by("l_shipmode").agg(
            n=bt.count(),
            total=col("l_quantity").sum(),
            low=col("l_quantity").min(),
            high=col("l_quantity").max(),
            sketch=bt.approx_n_unique(col("l_partkey")),
        )
        for shard in shards
    ]
    merged = partials[0]
    for piece in partials[1:]:
        merged = merged.union(piece)

    combined = (
        merged.group_by("l_shipmode")
        .agg(
            n=col("n").sum(),
            total=col("total").sum(),
            low=col("low").min(),
            high=col("high").max(),
        )
        .sort("l_shipmode")
        .to_pydict()
    )

    reference = (
        lineitem.group_by("l_shipmode")
        .agg(
            n=bt.count(),
            total=col("l_quantity").sum(),
            low=col("l_quantity").min(),
            high=col("l_quantity").max(),
        )
        .sort("l_shipmode")
        .to_pydict()
    )

    for column in ("n", "total", "low", "high"):
        assert combined[column] == reference[column], column
    print("sum, count, min and max all fold correctly across four shards")

    # Unbounded state: a mean folds only if you carry the pieces.
    mean_from_pieces = [
        total / count for total, count in zip(combined["total"], combined["n"], strict=True)
    ]
    true_mean = (
        lineitem.group_by("l_shipmode")
        .agg(m=col("l_quantity").mean())
        .sort("l_shipmode")
        .to_pydict()["m"]
    )
    assert all(abs(a - b) < 1e-9 for a, b in zip(mean_from_pieces, true_mean, strict=True))
    print("a mean folds when carried as (sum, count)")

    # A median does not fold at all: the per-shard medians do not combine.
    shard_medians = [
        shard.agg(m=bt.median(col("l_quantity"))).to_pydict()["m"][0] for shard in shards
    ]
    whole_median = lineitem.agg(m=bt.median(col("l_quantity"))).to_pydict()["m"][0]
    naive = sum(shard_medians) / len(shard_medians)
    print(f"median of the whole {whole_median}, mean of shard medians {naive}")
    assert isinstance(whole_median, float | int)


if __name__ == "__main__":
    main()
