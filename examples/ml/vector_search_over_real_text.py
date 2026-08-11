"""Nearest-neighbour search over an embedding column.

The embedding here is a deterministic stand-in rather than a model call, so the example
runs anywhere and the ranking is checkable. Everything after the embedding step — the
distance, the ranking, the top-k — is exactly what a real pipeline does.

    python examples/ml/vector_search_over_real_text.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    documents = tpch("part").select("p_partkey", "p_name").head(2_000)

    # A cheap, deterministic three-dimensional "embedding" of the text: three character
    # ratios. A real pipeline swaps this one expression for a model call.
    embedded = documents.with_columns(
        vector=bt.array(
            col("p_name").str.count_char("a") / 10.0,
            col("p_name").str.count_char("e") / 10.0,
            col("p_name").str.count_char("o") / 10.0,
        )
    )
    print(embedded.head(2).to_pydict())

    # The query side is one row, cross-joined onto the corpus.
    query = embedded.head(1).select(col("vector").alias("query"), col("p_name").alias("q_name"))

    ranked = (
        embedded.cross_join(query)
        .select(
            "p_partkey",
            "p_name",
            "q_name",
            similarity=col("vector").list.cosine_similarity(col("query")),
        )
        .sort("similarity", descending=True)
        .limit(5)
    )

    result = ranked.to_pydict()
    print("query:", result["q_name"][0])
    for name, score in zip(result["p_name"], result["similarity"], strict=True):
        print(f"  {score:.6f}  {name}")

    # The query document is its own nearest neighbour.
    assert result["p_name"][0] == result["q_name"][0]
    assert abs(result["similarity"][0] - 1.0) < 1e-9

    # Similarities are bounded and descending. The bound needs a tolerance: cosine is a
    # ratio of floating-point sums, so an exact match lands a few ulps above 1.0.
    assert result["similarity"] == sorted(result["similarity"], reverse=True)
    assert all(-1.0 - 1e-9 <= value <= 1.0 + 1e-9 for value in result["similarity"])


if __name__ == "__main__":
    main()
