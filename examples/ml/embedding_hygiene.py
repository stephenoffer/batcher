"""Checking an embedding column before you index it.

Zero vectors, duplicates and wrong dimensions are the three failures that make a vector
index quietly useless. All three are one aggregate each, and finding them before the index
is built is the difference between a bad build and a bad quarter.

    python examples/ml/embedding_hygiene.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    documents = tpch("part").select("p_partkey", "p_name").head(3_000)

    embedded = documents.with_columns(
        vector=bt.array(
            col("p_name").str.count_char("a") / 10.0,
            col("p_name").str.count_char("e") / 10.0,
            col("p_name").str.count_char("i") / 10.0,
        )
    )

    health = embedded.select(
        "p_partkey",
        dim=col("vector").list.dim(),
        magnitude=col("vector").list.magnitude(),
        zero=col("vector").list.is_zero_vector(),
    )

    summary = health.agg(
        rows=bt.count(),
        zeros=bt.count_if(col("zero")),
        smallest=col("magnitude").min(),
        dims=col("dim").n_unique(),
    ).to_pydict()
    print(summary)

    # Every vector has the same dimension — a mixed-dimension column cannot be indexed.
    assert summary["dims"][0] == 1

    # Zero vectors carry no direction, so cosine similarity against them is undefined.
    zero_count = summary["zeros"][0]
    print(f"{zero_count} zero vectors of {summary['rows'][0]}")
    assert zero_count == health.filter(col("magnitude") == 0.0).count()

    # Drop them before indexing rather than after.
    usable = health.filter(~col("zero"))
    assert usable.count() == summary["rows"][0] - zero_count
    assert usable.agg(m=col("magnitude").min()).to_pydict()["m"][0] > 0

    # Duplicate vectors are wasted index space and identical search results.
    vectors = embedded.select(key=col("vector").list.join(",")).to_pydict()["key"]
    distinct = len(set(vectors))
    print(f"{distinct} distinct vectors of {len(vectors)}")
    assert distinct <= len(vectors)


if __name__ == "__main__":
    main()
