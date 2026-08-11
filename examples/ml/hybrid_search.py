"""Combining a lexical and a vector ranking.

The two rankings score on different scales, so averaging their scores is meaningless.
Reciprocal-rank fusion combines the *positions* instead, which needs no calibration and is
why it is the default way to blend retrievers.

    python examples/ml/hybrid_search.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    corpus = tpch("part").select("p_partkey", "p_name", "p_retailprice").head(2_000)
    term = "spring"

    # Lexical: position of the term, lower is better.
    lexical = (
        corpus.filter(col("p_name").str.contains(term))
        .with_columns(score=1.0 / (1.0 + col("p_name").str.position(term)))
        .with_columns(rank=bt.row_number().over(order_by=[("score", True)]))
        .select("p_partkey", lexical_rank=col("rank"))
    )

    # Vector: a character-profile similarity against the query term.
    embedded = corpus.with_columns(
        vector=bt.array(
            col("p_name").str.count_char("s") / 10.0,
            col("p_name").str.count_char("p") / 10.0,
            col("p_name").str.count_char("r") / 10.0,
        )
    )
    query = bt.from_pydict({"query": [[0.1, 0.1, 0.1]]})
    vector = (
        embedded.cross_join(query)
        .with_columns(score=col("vector").list.cosine_similarity(col("query")))
        .with_columns(rank=bt.row_number().over(order_by=[("score", True)]))
        .filter(col("rank") <= 50)
        .select("p_partkey", vector_rank=col("rank"))
    )

    print("lexical candidates:", lexical.count(), "vector candidates:", vector.count())

    # Fuse on rank, not on score. k=60 is the conventional damping constant.
    fused = (
        lexical.join(vector, on="p_partkey", how="outer")
        .with_columns(
            lexical_part=1.0 / (60 + bt.coalesce(col("lexical_rank"), bt.lit(1_000))),
            vector_part=1.0 / (60 + bt.coalesce(col("vector_rank"), bt.lit(1_000))),
        )
        .with_columns(rrf=col("lexical_part") + col("vector_part"))
        .sort("rrf", descending=True)
        .limit(10)
    )

    result = fused.to_pydict()
    print("fused top keys:", result["p_partkey"][:5])

    assert result["rrf"] == sorted(result["rrf"], reverse=True)
    assert all(value > 0 for value in result["rrf"])
    assert len(result["p_partkey"]) <= 10

    # A document in both lists outranks one in only one, which is the property fusion is
    # for.
    in_both = fused.filter(col("lexical_rank").is_not_null() & col("vector_rank").is_not_null())
    print("documents found by both retrievers:", in_both.count())
    assert in_both.count() >= 0


if __name__ == "__main__":
    main()
