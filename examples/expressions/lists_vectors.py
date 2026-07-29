"""Embedding vectors as list columns: similarity, distance, and normalization.

A list column of floats is an embedding. Keeping it in the engine means a similarity
search is a projection plus a sort rather than a round trip through NumPy, and it stays
columnar when the table is larger than memory.

    python examples/expressions/lists_vectors.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    # Three 3-d vectors. `a` and `b` point the same way; `c` is orthogonal to `a`.
    vectors = bt.from_pydict(
        {
            "id": ["a", "b", "c"],
            "vec": [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "query": [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        }
    )

    scored = vectors.with_columns(
        # Direction-only similarity: 1.0 means identical direction.
        cosine=col("vec").list.cosine_similarity(col("query")),
        cos_dist=col("vec").list.cosine_distance(col("query")),
        dot=col("vec").list.dot(col("query")),
        # Metric distances.
        l2=col("vec").list.l2_distance(col("query")),
        l1=col("vec").list.l1_distance(col("query")),
        euclid=col("vec").list.euclidean_distance(col("query")),
        # Vector properties.
        dim=col("vec").list.dim(),
        norm=col("vec").list.magnitude(),
        l2_norm=col("vec").list.l2_norm(),
        unit=col("vec").list.is_unit_norm(),
        zero=col("vec").list.is_zero_vector(),
        normalized=col("vec").list.normalize(),
    )

    result = scored.to_pydict()
    print(result)

    # `b` is `a` scaled by 2, so cosine similarity is identical (1.0) but L2 distance is not.
    assert result["cosine"][0] == 1.0
    assert result["cosine"][1] == 1.0
    assert result["cos_dist"][0] == 0.0
    # `c` is orthogonal to the query.
    assert result["cosine"][2] == 0.0
    assert result["dot"] == [1.0, 2.0, 0.0]
    assert result["l2"][:2] == [0.0, 1.0]
    assert abs(result["l2"][2] - 2.0**0.5) < 1e-9
    assert result["dim"] == [3, 3, 3]
    assert result["norm"] == [1.0, 2.0, 1.0]
    assert result["l2_norm"] == result["norm"]
    assert result["unit"] == [True, False, True]
    assert result["zero"] == [False, False, False]
    # Normalizing `b` gives back the unit vector `a`.
    assert result["normalized"][1] == [1.0, 0.0, 0.0]

    # The search this exists for: rank by similarity to the query.
    ranked = (
        vectors.select(id=col("id"), score=col("vec").list.cosine_similarity(col("query")))
        .sort("score", descending=True)
        .to_pydict()
    )
    print(ranked)
    assert ranked["id"][-1] == "c"


if __name__ == "__main__":
    main()
