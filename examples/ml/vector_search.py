"""Vector search over an embedding column, in the engine.

Keeping vectors as a list column means retrieval is a projection plus a top-N, composable
with any other filter. That is what lets you pre-filter by metadata *before* scoring,
which is both faster and more correct than scoring everything and filtering after.

    python examples/ml/vector_search.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    # A tiny corpus with 3-d "embeddings" and some metadata to filter on.
    corpus = bt.from_pydict(
        {
            "doc_id": ["d1", "d2", "d3", "d4"],
            "lang": ["en", "en", "fr", "en"],
            "vec": [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
        }
    )
    # There is no list literal, so carry the query vector as a broadcast column. In a
    # real service this is the one row you got back from the embedding call.
    query = [1.0, 0.0, 0.0]
    n = corpus.count()
    with_query = corpus.cross_join(bt.from_pydict({"q": [query]}))
    assert with_query.count() == n

    # Score every row against the query vector.
    scored = with_query.with_columns(
        score=col("vec").list.cosine_similarity(col("q")),
        distance=col("vec").list.cosine_distance(col("q")),
    ).to_pydict()
    print(scored["score"])

    assert scored["score"][0] == 1.0
    assert scored["score"][3] == 0.0
    # Similarity and distance are complements.
    assert all(
        abs(s + d - 1.0) < 1e-9 for s, d in zip(scored["score"], scored["distance"], strict=True)
    )

    # Top-k retrieval: a heap, not a full sort.
    top = (
        with_query.with_columns(score=col("vec").list.cosine_similarity(col("q")))
        .top_k(2, by="score")
        .to_pydict()
    )
    print("top 2:", top["doc_id"], top["score"])
    assert set(top["doc_id"]) == {"d1", "d3"}

    # Pre-filtering by metadata before scoring: fewer vectors touched, and the language
    # constraint is honoured exactly rather than approximately.
    english = (
        with_query.filter(col("lang") == "en")
        .with_columns(score=col("vec").list.cosine_similarity(col("q")))
        .sort("score", descending=True)
        .limit(2)
        .to_pydict()
    )
    print("english top 2:", english["doc_id"])
    assert english["doc_id"] == ["d1", "d2"]
    assert "d3" not in english["doc_id"]  # the French doc is excluded, not just outranked

    # A similarity threshold instead of a fixed k, when recall matters more than count.
    relevant = (
        with_query.with_columns(score=col("vec").list.cosine_similarity(col("q")))
        .filter(col("score") > 0.8)
        .to_pydict()
    )
    assert sorted(relevant["doc_id"]) == ["d1", "d2", "d3"]

    # Health checks on the embedding column itself.
    health = corpus.select(
        zero_rate=bt.zero_vector_rate("vec"), unit_rate=bt.unit_norm_rate("vec")
    ).to_pydict()
    print("embedding health:", health)
    assert health["zero_rate"][0] == 0.0


if __name__ == "__main__":
    main()
