"""Explode, transform, regroup: the round trip through a flat relation.

Some operations are easier one element per row. Exploding, doing them, and collecting back is
a legitimate shape — as long as you carry the key that lets you regroup, and check the counts
on the way back.

    python examples/expr_collections/exploding_and_regrouping.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    per_order = (
        tpch("lineitem")
        .head(20_000)
        .group_by("l_orderkey")
        .agg(parts=bt.array_agg(col("l_partkey")))
        .sort("l_orderkey")
    )
    before = per_order.select("l_orderkey", n=col("parts").list.len()).to_pydict()
    print("orders:", per_order.count(), "elements:", sum(before["n"]))

    # Explode, carrying the key.
    flat = per_order.explode("parts").select("l_orderkey", part=col("parts"))
    assert flat.count() == sum(before["n"])

    # Do the per-element work.
    transformed = flat.with_columns(bucket=col("part") % 10)

    # Regroup on the key.
    after = (
        transformed.group_by("l_orderkey")
        .agg(parts=bt.array_agg(col("part")), buckets=bt.array_agg(col("bucket")))
        .sort("l_orderkey")
    )
    counts = after.select("l_orderkey", n=col("parts").list.len()).to_pydict()

    # Same keys, same list lengths — nothing was lost or duplicated.
    assert counts["l_orderkey"] == before["l_orderkey"]
    assert counts["n"] == before["n"]

    # The per-element work really happened.
    sample = after.head(1).to_pydict()
    assert all(
        bucket == part % 10
        for part, bucket in zip(sample["parts"][0], sample["buckets"][0], strict=True)
    )
    print("first order:", sample["parts"][0], "->", sample["buckets"][0])

    # An order whose list was empty would vanish in the explode, which is the one thing to
    # check for: here every order has at least one line, so none do.
    assert min(before["n"]) >= 1
    assert after.count() == per_order.count()


if __name__ == "__main__":
    main()
