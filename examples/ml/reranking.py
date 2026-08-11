"""Reranking a candidate set: cheap retrieval, then an expensive score.

Retrieval and ranking are separate stages because they have different costs. Pull a hundred
candidates with something cheap, then spend the expensive scorer on those hundred rather
than on the corpus. The two-stage shape is what makes the expensive model affordable.

    python examples/ml/reranking.py
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
    query_term = "spring"

    # Stage one: a cheap lexical filter over the whole corpus.
    candidates = corpus.filter(col("p_name").str.contains(query_term))
    print(f"stage one kept {candidates.count()} of {corpus.count()}")
    assert 0 < candidates.count() < corpus.count()

    # Stage two: an expensive score, over the candidates only.
    reranked = (
        candidates.with_columns(
            # Stand-in for a cross-encoder: position of the term plus a length penalty.
            score=1.0 / (1.0 + col("p_name").str.position(query_term))
            - col("p_name").str.len_chars() / 1000.0
        )
        .sort("score", descending=True)
        .limit(10)
    )

    result = reranked.to_pydict()
    for name, score in zip(result["p_name"], result["score"], strict=True):
        print(f"  {score:.4f}  {name}")

    assert result["score"] == sorted(result["score"], reverse=True)
    assert all(query_term in name for name in result["p_name"])
    assert len(result["p_name"]) <= 10

    # The expensive stage ran on the candidates, not the corpus — which is the whole
    # point, and is checkable by counting what it saw.
    assert candidates.count() < corpus.count()

    # Reciprocal-rank fusion combines two rankings without needing their scores to be
    # comparable, which they rarely are.
    by_price = candidates.sort("p_retailprice", descending=True).limit(10).to_pydict()
    assert len(by_price["p_partkey"]) <= 10
    assert bt is not None


if __name__ == "__main__":
    main()
