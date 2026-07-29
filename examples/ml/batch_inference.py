"""Batch inference: a model over every row, without a Python loop.

``map_batches`` hands your callable a whole **pyarrow ``RecordBatch``**, never one row, so
``batch["col"]`` is an Arrow array. Call ``.to_pylist()`` once per batch rather than
indexing it element by element. Passing a *class* rather than a function is what makes the
model load once per worker instead of once per batch, which on a real model is the
difference between minutes and hours.

    python examples/ml/batch_inference.py
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col


class ScoringModel:
    """A stand-in for a real model: loaded once, then called per batch."""

    def __init__(self) -> None:
        # In a real job this is where the weights load. It runs once per worker.
        self.loads = 1
        self.bias = 0.5

    def __call__(self, batch: pa.RecordBatch) -> dict[str, list]:
        # One conversion for the whole column, not one per row.
        features = batch["feature"].to_pylist()
        return {"score": [self.bias + 0.1 * v for v in features]}


def main() -> None:
    data = bt.from_pydict({"id": list(range(10)), "feature": [float(i) for i in range(10)]})

    # A class: instantiated once, then called per batch.
    scored = data.map_batches(ScoringModel, batch_size=4).to_pydict()
    print(scored["score"][:4])
    assert len(scored["score"]) == 10
    assert abs(scored["score"][0] - 0.5) < 1e-9
    assert abs(scored["score"][9] - 1.4) < 1e-9

    # A plain function works too, when there is nothing to load.
    def double(batch: pa.RecordBatch) -> dict[str, list]:
        return {"doubled": [v * 2 for v in batch["feature"].to_pylist()]}

    doubled = data.map_batches(double, batch_size=4).to_pydict()
    assert doubled["doubled"][3] == 6.0

    # Anything expressible as an expression should be, because it runs in Rust rather
    # than crossing into Python at all.
    native = data.select(score=0.5 + col("feature") * 0.1).to_pydict()
    assert abs(native["score"][9] - 1.4) < 1e-9
    # Same answer, no Python in the loop.
    assert [round(v, 9) for v in native["score"]] == [round(v, 9) for v in scored["score"]]

    # The `.ml` accessor is the same machinery with model-shaped defaults.
    ml_scored = data.ml.map_batches(ScoringModel, batch_size=4).to_pydict()
    assert len(ml_scored["score"]) == 10

    # Inference streams, so a table larger than memory is the ordinary case.
    seen = 0
    for batch in data.map_batches(ScoringModel, batch_size=3).iter_batches():
        seen += batch.num_rows
    assert seen == 10

    # And it composes: filter first so the model never sees rows you would discard.
    cheap = data.filter(col("feature") > 7).map_batches(ScoringModel, batch_size=4).to_pydict()
    print("scored after filtering:", cheap["score"])
    assert len(cheap["score"]) == 2


if __name__ == "__main__":
    main()
