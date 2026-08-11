"""Distances between embedding vectors held in a list column.

The query vector is a *column*, not a literal — these functions compare two list columns,
so a one-row query dataset cross-joined onto the corpus is the shape that works. That is
also the shape a real search takes, where the query side is a batch of queries rather than
one.

Cosine similarity is scale-free, so it is the one to use when the vectors are not
normalized. L2 and cosine agree on ranking only when they are.

    python examples/expr_vectors/distance_measures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from batcher import col


def main() -> None:
    corpus = bt.from_pydict(
        {
            "id": [1, 2, 3, 4],
            "vector": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
        }
    )
    query = bt.from_pydict({"query": [[1.0, 0.0, 0.0]]})

    scored = corpus.cross_join(query).select(
        "id",
        cosine=col("vector").list.cosine_similarity(col("query")),
        cosine_distance=col("vector").list.cosine_distance(col("query")),
        l2=col("vector").list.l2_distance(col("query")),
        l1=col("vector").list.l1_distance(col("query")),
        dot=col("vector").list.dot(col("query")),
        magnitude=col("vector").list.magnitude(),
    )

    result = scored.sort("id").to_pydict()
    print(
        {
            name: [round(value, 4) for value in column]
            for name, column in result.items()
            if name != "id"
        }
    )

    # Vector 1 is the query itself.
    assert abs(result["cosine"][0] - 1.0) < 1e-9
    assert abs(result["l2"][0]) < 1e-9

    # Vector 2 is orthogonal to it; vector 3 points the same way but is twice as long,
    # which cosine calls identical and L2 does not.
    assert abs(result["cosine"][1]) < 1e-9
    assert abs(result["cosine"][2] - 1.0) < 1e-9
    assert result["l2"][2] > 0.0

    # Cosine distance is one minus cosine similarity.
    assert all(
        abs(distance - (1.0 - similarity)) < 1e-9
        for similarity, distance in zip(result["cosine"], result["cosine_distance"], strict=True)
    )

    # Nearest neighbours by cosine: the two collinear vectors tie at the top.
    ranked = scored.sort("cosine", descending=True).to_pydict()
    print("nearest:", ranked["id"][:2])
    assert set(ranked["id"][:2]) == {1, 3}


if __name__ == "__main__":
    main()
