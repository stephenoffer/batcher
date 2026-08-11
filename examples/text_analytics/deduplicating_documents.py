"""Finding near-duplicate text with a similarity hash.

Exact deduplication is a group-by on the text. Near-duplicate detection needs a hash where
similar inputs collide, which is what MinHash is for — and it is an ordinary column
expression here rather than a separate library.

    python examples/text_analytics/deduplicating_documents.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    comments = tpch("lineitem").select("l_orderkey", "l_comment").head(20_000)

    # Exact duplicates: a group-by on the text itself.
    exact = (
        comments.group_by("l_comment")
        .agg(copies=bt.count())
        .filter(col("copies") > 1)
        .sort("copies", descending=True)
    )
    print("exactly duplicated comments:", exact.count())

    total = comments.count()
    distinct = comments.n_unique("l_comment")
    print(f"{distinct} distinct of {total}")
    assert distinct <= total

    # A content hash gives the same grouping in fixed width, which is what you store when
    # the text is large.
    hashed = comments.select("l_orderkey", fingerprint=col("l_comment").str.sha256())
    assert hashed.n_unique("fingerprint") == distinct

    # MinHash is the near-duplicate version: similar text, colliding signatures.
    similar = comments.select(
        "l_orderkey",
        sim=col("l_comment").str.minhash(),
    )
    buckets = (
        similar.group_by("sim").agg(n=bt.count()).filter(col("n") > 1).sort("n", descending=True)
    )
    print("minhash buckets with more than one member:", buckets.count())

    # A hash collision groups at least as aggressively as exact equality does.
    assert similar.n_unique("sim") <= distinct


if __name__ == "__main__":
    main()
