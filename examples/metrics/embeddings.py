"""Corpus-level embedding metrics: monitoring a vector column in aggregate.

Per-row similarity is a projection; these are the aggregates over it. They are the cheap
health checks for an embedding job: a drifting mean cosine similarity or a rising
zero-vector rate usually means the upstream text changed, not the model.

    python examples/metrics/embeddings.py
"""

from __future__ import annotations

import batcher as bt


def main() -> None:
    pairs = bt.from_pydict(
        {
            "left": [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
            "right": [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        }
    )

    agg = pairs.select(
        cos_sim=bt.mean_cosine_similarity("left", "right"),
        cos_dist=bt.mean_cosine_distance("left", "right"),
        angular=bt.mean_angular_distance("left", "right"),
        dot=bt.mean_dot_product("left", "right"),
        euclid=bt.mean_euclidean_distance("left", "right"),
        manhattan=bt.mean_manhattan_distance("left", "right"),
        norm=bt.mean_embedding_norm("left"),
        unit_rate=bt.unit_norm_rate("left"),
        zero_rate=bt.zero_vector_rate("left"),
    ).to_pydict()

    print(agg)

    # Two identical pairs (1.0) and one orthogonal pair (0.0) -> mean 2/3.
    assert abs(agg["cos_sim"][0] - 2 / 3) < 1e-12
    assert abs(agg["cos_dist"][0] - 1 / 3) < 1e-12
    assert abs(agg["dot"][0] - 2 / 3) < 1e-12
    # Every left vector is already unit length.
    assert agg["norm"] == [1.0]
    assert agg["unit_rate"] == [1.0]
    assert agg["zero_rate"] == [0.0]

    # The health check this exists for: a zero vector is a failed embedding call.
    broken = bt.from_pydict({"v": [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]})
    rate = broken.select(zero=bt.zero_vector_rate("v")).to_pydict()
    print(rate)
    assert abs(rate["zero"][0] - 2 / 3) < 1e-12


if __name__ == "__main__":
    main()
