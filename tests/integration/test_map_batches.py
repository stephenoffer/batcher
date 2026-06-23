"""map_batches — opaque Python/ML operators composed with relational ops.

This is the batch-inference / embedding path: arbitrary Python (a model) runs over
Arrow batches, freely interleaved with compiled relational operators.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import batcher as bt
from batcher import col

pytest.importorskip("batcher._native", reason="native engine not built")


def test_map_batches_inference_pipeline():
    def fake_embed(batch: pa.RecordBatch) -> pa.RecordBatch:
        x = batch.column("x").to_numpy()
        y = batch.column("y").to_numpy()
        return batch.append_column("emb", pa.array(x * 0.5 + y))

    out = (
        bt.from_pydict({"x": list(range(10)), "y": list(range(100, 110))})
        .filter(col("x") >= 5)
        .map_batches(fake_embed, output_columns=["x", "y", "emb"])
        .select("x", "emb")
        .collect()
    )
    d = out.to_pydict()
    assert d["x"] == [5, 6, 7, 8, 9]
    assert d["emb"] == [5 * 0.5 + 105, 6 * 0.5 + 106, 7 * 0.5 + 107, 8 * 0.5 + 108, 9 * 0.5 + 109]


def test_map_batches_rebatches_to_batch_size():
    seen: list[int] = []

    def spy(batch: pa.RecordBatch) -> pa.RecordBatch:
        seen.append(batch.num_rows)
        return batch

    bt.from_pydict({"x": list(range(100))}).map_batches(spy, batch_size=32).collect()
    assert seen == [32, 32, 32, 4]


def test_map_batches_composes_with_aggregate():
    def inc(b: pa.RecordBatch) -> pa.RecordBatch:
        return b.set_column(1, "v", pc.add(b.column("v"), 1))

    out = (
        bt.from_pydict({"g": ["a", "a", "b"], "v": [1, 2, 3]})
        .map_batches(inc, output_columns=["g", "v"])
        .group_by("g")
        .agg(s=col("v").sum())
        .collect()
        .to_pydict()
    )
    assert dict(zip(out["g"], out["s"], strict=True)) == {"a": 5, "b": 4}


def test_map_batches_dict_return():
    out = (
        bt.from_pydict({"x": [1, 2, 3]})
        .map_batches(
            lambda b: {"x2": [v * v for v in b.column("x").to_pylist()]}, output_columns=["x2"]
        )
        .collect()
        .to_pydict()
    )
    assert out == {"x2": [1, 4, 9]}
