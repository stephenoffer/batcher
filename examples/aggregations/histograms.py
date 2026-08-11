"""Binning a column: the histogram aggregate and explicit width buckets.

`bt.histogram` returns (value, count) pairs for a discrete column. For a continuous one
you want fixed-width bins instead, which is what `width_bucket` produces — and then the
binning is a plain group-by on the bucket number.

    python examples/aggregations/histograms.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem")

    # Discrete column: one entry per distinct value.
    counts = lineitem.agg(shape=bt.histogram(col("l_linenumber"))).to_pydict()["shape"][0]
    print("line-number histogram:", counts)
    assert sum(count for _, count in counts) == lineitem.count()

    # Continuous column: fixed-width buckets between two bounds.
    binned = (
        lineitem.with_columns(bucket=bt.width_bucket(col("l_extendedprice"), 0.0, 100_000.0, 10))
        .group_by("bucket")
        .agg(lines=bt.count(), lowest=col("l_extendedprice").min())
        .sort("bucket")
        .to_pydict()
    )
    print(binned["bucket"], binned["lines"])

    # Every row lands in exactly one bucket.
    assert sum(binned["lines"]) == lineitem.count()
    # Buckets are ordered, so their minimum values are too.
    assert binned["lowest"] == sorted(binned["lowest"])


if __name__ == "__main__":
    main()
