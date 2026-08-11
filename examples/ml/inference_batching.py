"""Batching an expensive per-call model so the overhead amortizes.

The cost of a Python callback is paid per batch, not per row, so the batch size is the knob
that decides how much of it there is. Measuring calls against rows is the check that the
callback is genuinely batch-first.

    python examples/ml/inference_batching.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow as pa

from _common import tpch


class CountingModel:
    """Records how often it was called, which is the thing under test."""

    def __init__(self) -> None:
        self.calls = 0
        self.rows = 0

    def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        self.calls += 1
        self.rows += batch.num_rows
        prices = batch.column("l_extendedprice").to_pylist()
        scored = [value * 0.001 for value in prices]
        return pa.RecordBatch.from_arrays(
            [*batch.columns, pa.array(scored, type=pa.float64())],
            names=[*batch.schema.names, "score"],
        )


def main() -> None:
    lineitem = tpch("lineitem").select("l_orderkey", "l_extendedprice")
    total = lineitem.count()

    model = CountingModel()
    result = lineitem.map_batches(model).to_pydict()

    print(f"{model.calls} calls for {model.rows} rows")
    assert model.rows == total
    assert "score" in result
    assert len(result["score"]) == total

    # The whole point: far fewer calls than rows.
    assert model.calls < total / 100
    print(f"rows per call: {model.rows / model.calls:.0f}")

    # The scores are right, which matters more than the batching.
    assert all(
        abs(score - price * 0.001) < 1e-9
        for price, score in zip(result["l_extendedprice"], result["score"], strict=True)
    )

    # A larger batch means fewer calls for the same work.
    coarse = CountingModel()
    consumed = 0
    for batch in lineitem.iter_batches(batch_size=65_536):
        coarse(batch)
        consumed += batch.num_rows
    print(f"at 65,536 rows per batch: {coarse.calls} calls")
    assert consumed == total
    assert coarse.calls <= model.calls


if __name__ == "__main__":
    main()
