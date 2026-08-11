"""Joining two sets of vectors by similarity rather than by key.

A similarity join is a cross join plus a distance plus a top-k per left row. Written that
way it is obviously quadratic, which is the honest starting point: it works up to a size,
and past that you need an index.

    python examples/ml/similarity_join.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    parts = tpch("part").select("p_partkey", "p_name").head(300)

    def embed(dataset: bt.Dataset, prefix: str) -> bt.Dataset:
        return dataset.select(
            col("p_partkey").alias(f"{prefix}_key"),
            col("p_name").alias(f"{prefix}_name"),
            bt.array(
                col("p_name").str.count_char("a") / 10.0,
                col("p_name").str.count_char("e") / 10.0,
                col("p_name").str.count_char("i") / 10.0,
            ).alias(f"{prefix}_vector"),
        )

    left = embed(parts.head(20), "q")
    right = embed(parts, "d")

    scored = left.cross_join(right).select(
        "q_key",
        "q_name",
        "d_key",
        "d_name",
        score=col("q_vector").list.cosine_similarity(col("d_vector")),
    )
    print("candidate pairs:", scored.count())
    assert scored.count() == left.count() * right.count()

    top_three = scored.with_columns(
        rank=bt.row_number().over(partition_by=["q_key"], order_by=[("score", True)])
    ).filter(col("rank") <= 3)

    print("kept pairs:", top_three.count())
    assert top_three.count() <= left.count() * 3

    # Every query keeps at most three neighbours, and the best is itself.
    per_query = top_three.group_by("q_key").agg(kept=bt.count()).to_pydict()
    assert all(value <= 3 for value in per_query["kept"])

    best = top_three.filter(col("rank") == 1).to_pydict()
    self_matches = sum(1 for q, d in zip(best["q_key"], best["d_key"], strict=True) if q == d)
    print(f"{self_matches} of {len(best['q_key'])} queries matched themselves first")
    assert self_matches > 0


if __name__ == "__main__":
    main()
