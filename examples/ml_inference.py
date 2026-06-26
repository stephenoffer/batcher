"""Batch inference: score every row with a model-shaped callable.

The same contract `ds.ml.infer` runs — a callable applied once per Arrow batch —
but with a plain NumPy scoring function instead of a real model, so it runs
anywhere with no GPU, weights, or extra dependency. Swap ``LinearScorer`` for a
class that loads a checkpoint in ``__init__`` and runs a forward pass in
``__call__`` and the pipeline is unchanged.

A Dataset is lazy and immutable: each step returns a new Dataset and nothing
executes until the terminal ``to_pydict``.

Run it directly::

    python examples/ml_inference.py
"""

from __future__ import annotations

import batcher as bt


class LinearScorer:
    """A model-shaped callable: set up once, called per batch.

    Passing a *class* (not a function) is the inference pattern — the engine
    instantiates it once per worker, so weights load a single time and every
    batch reuses them. ``batch_format="numpy"`` hands ``__call__`` a
    ``{column: ndarray}`` dict and accepts one back.
    """

    def __init__(self, weights: dict[str, float], bias: float, threshold: float) -> None:
        self.weights = weights
        self.bias = bias
        self.threshold = threshold

    def __call__(self, batch: dict) -> dict:
        import numpy as np

        logit = np.full(len(next(iter(batch.values()))), self.bias, dtype="float64")
        for name, weight in self.weights.items():
            logit = logit + weight * batch[name]
        score = 1.0 / (1.0 + np.exp(-logit))  # sigmoid
        batch["score"] = score
        batch["label"] = (score >= self.threshold).astype("int64")
        return batch


def main() -> None:
    features = bt.from_pydict(
        {
            "id": [1, 2, 3, 4, 5],
            "recency": [0.9, 0.1, 0.6, 0.2, 0.8],
            "frequency": [0.7, 0.2, 0.5, 0.1, 0.9],
        }
    )

    scorer = LinearScorer(
        weights={"recency": 3.0, "frequency": 2.0},
        bias=-2.5,
        threshold=0.5,
    )

    # Lazy: the scorer runs only when the terminal to_pydict() executes. A real
    # model adds num_gpus=/concurrency= here to place it on a GPU actor pool.
    # output_columns declares the schema the stage adds, so later ops can name them.
    scored = features.ml.map_batches(
        scorer,
        batch_format="numpy",
        output_columns=["id", "recency", "frequency", "score", "label"],
    )

    # Keep only the rows the model flags, ordered by confidence — ordinary
    # relational ops compose with the inference stage.
    flagged = scored.filter(bt.col("label") == 1).sort("score", descending=True)

    result = flagged.to_pydict()
    print(result)

    # Rows 1, 3, 5 clear the 0.5 threshold; row 5 scores highest.
    assert result["id"] == [5, 1, 3]
    assert all(label == 1 for label in result["label"])
    assert result["score"] == sorted(result["score"], reverse=True)
    assert all(0.5 <= s <= 1.0 for s in result["score"])

    # The full table still carries a score for every input row.
    everyone = scored.to_pydict()
    assert len(everyone["id"]) == 5
    assert set(everyone["label"]) == {0, 1}


if __name__ == "__main__":
    main()
