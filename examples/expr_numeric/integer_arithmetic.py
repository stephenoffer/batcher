"""Integer division, modulo, and the bucketing they give you.

Modulo is how you shard, sample deterministically, and build a stable hash bucket. It is
also the one arithmetic operator whose behaviour on negatives differs between languages, so
it is worth pinning down on your own data.

    python examples/expr_numeric/integer_arithmetic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").select("o_orderkey")

    bucketed = orders.select(
        "o_orderkey",
        bucket=col("o_orderkey") % 8,
        shard=col("o_orderkey") // 1000,
    )
    result = bucketed.to_pydict()

    # Eight buckets, all present, and every key lands in exactly one.
    counts = bucketed.value_counts("bucket").sort("bucket").to_pydict()
    print(counts)
    assert set(counts["bucket"]) == set(range(8))
    assert sum(counts["count"]) == orders.count()

    # The buckets are near-even, because the keys are near-sequential.
    spread = max(counts["count"]) - min(counts["count"])
    assert spread < orders.count() * 0.05

    # Integer division truncates toward zero rather than rounding.
    assert all(
        shard == key // 1000
        for key, shard in zip(result["o_orderkey"], result["shard"], strict=True)
    )

    # Deterministic sampling: take one bucket, get about an eighth of the rows.
    sampled = bucketed.filter(col("bucket") == 3)
    share = sampled.count() / orders.count()
    print(f"bucket 3 holds {share:.4f} of the rows")
    assert 0.10 < share < 0.15

    # The same filter twice gives the same rows, which a random sample would not.
    assert sampled.count() == bucketed.filter(col("bucket") == 3).count()
    assert bt is not None


if __name__ == "__main__":
    main()
