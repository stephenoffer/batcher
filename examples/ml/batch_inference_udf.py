"""Batch inference with a plain Python callable.

`map_batches` hands your function a whole Arrow batch, never a row. That is the contract
that keeps a Python model call from costing a Python function call per row — the overhead
amortizes over 16,384 rows instead of being paid 16,384 times.

    python examples/ml/batch_inference_udf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow as pa

from _common import tpch


class Scorer:
    """A class, so an expensive model loads once per worker rather than per batch."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.calls = 0

    def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        self.calls += 1
        prices = batch.column("l_extendedprice").to_pylist()
        labels = ["high" if value > self.threshold else "low" for value in prices]
        return pa.RecordBatch.from_arrays(
            [*batch.columns, pa.array(labels)],
            names=[*batch.schema.names, "band"],
        )


def main() -> None:
    lineitem = tpch("lineitem").select("l_orderkey", "l_extendedprice")

    scorer = Scorer(threshold=30_000.0)
    scored = lineitem.map_batches(scorer)

    result = scored.to_pydict()
    print(scorer.calls, "batches scored")
    assert scorer.calls > 1
    assert scorer.calls < lineitem.count()  # not once per row

    assert "band" in result
    assert len(result["band"]) == lineitem.count()
    assert set(result["band"]) == {"high", "low"}

    # The labels agree with the threshold that produced them.
    assert all(
        (label == "high") == (price > 30_000.0)
        for price, label in zip(result["l_extendedprice"], result["band"], strict=True)
    )

    # Where the logic can be expressed as an expression, it should be: no Python at all,
    # and the same answer.
    import batcher as bt
    from batcher import col

    native = lineitem.with_columns(
        band=bt.when(col("l_extendedprice") > 30_000.0)
        .then(bt.lit("high"))
        .otherwise(bt.lit("low"))
    ).to_pydict()
    assert native["band"] == result["band"]


if __name__ == "__main__":
    main()
