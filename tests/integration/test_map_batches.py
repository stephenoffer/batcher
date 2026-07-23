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


def test_map_batches_table_return_preserves_schema_on_empty_input():
    """A schema-changing UDF returning a *Table* (an allowed return type) must keep its
    output schema even when the input batch is empty (a filter removed every row), so a
    downstream reference to a UDF-added column works exactly as it does for a RecordBatch
    return. Previously a 0-row Table dropped to zero batches and the stage's schema fell
    back to the input's, so the downstream `select("y")` raised on empty input only."""

    def add_y_table(batch: pa.RecordBatch) -> pa.Table:
        t = pa.Table.from_batches([batch])
        return t.append_column("y", pc.add(t.column("x"), 100))

    empty = bt.from_pydict({"x": [1, 2, 3]}).filter(col("x") > 100)
    out = empty.map_batches(add_y_table, output_columns=["x", "y"]).select("y").collect()
    assert out.to_pydict() == {"y": []}

    # Non-empty still correct.
    nonempty = bt.from_pydict({"x": [1, 2, 3]})
    got = nonempty.map_batches(add_y_table, output_columns=["x", "y"]).select("y").collect()
    assert got.to_pydict() == {"y": [101, 102, 103]}


def test_stream_path_reconciles_schema_drift_matches_materializing():
    """A `map_batches` UDF whose output schema DRIFTS across batches must succeed on the
    streaming (GPU-inference) path exactly as it does on the materializing path.

    The streaming linear-chain path (taken for a `num_gpus > 0` stage) yields each stage's
    batches straight to the final `Table.from_batches`. When the UDF adds a column on only
    some batches, those batches carry differing schemas — the exact drift the materializing
    path unions with `reconcile_batches`. Previously the streaming path skipped that step, so
    the final concat raised `ArrowInvalid: Schema ... was different`, crashing a pipeline the
    staged path handled and contradicting the "identical to the staged materialization"
    contract. This asserts the streaming result now equals the materializing result."""

    def drift(batch: pa.RecordBatch) -> pa.RecordBatch:
        # Add column `c` only on batches whose first row is even, so batch_size=5 over
        # 0..19 yields batches [0-4]{x,c}, [5-9]{x}, [10-14]{x,c}, [15-19]{x} — drifting.
        if batch.column("x")[0].as_py() % 2 == 0:
            return batch.append_column("c", pc.multiply(batch.column("x"), 100))
        return batch

    data = {"x": list(range(20))}

    # Streaming path: a single GPU stage with an explicit batch_size is stream-eligible.
    # num_gpus>0 needs no real device here — autocast is a no-op on CPU.
    stream = (
        bt.from_pydict(data)
        .map_batches(drift, num_gpus=1, batch_size=5, num_workers=1, output_columns=["x", "c"])
        .collect()
        .to_pydict()
    )
    # Materializing path (num_gpus=0): the reference; it unions the drifted schema.
    mat = (
        bt.from_pydict(data)
        .map_batches(drift, batch_size=5, num_workers=1, output_columns=["x", "c"])
        .collect()
        .to_pydict()
    )
    assert stream == mat
    expected_c = [0, 100, 200, 300, 400, *([None] * 5), 1000, 1100, 1200, 1300, 1400, *([None] * 5)]
    assert stream["x"] == list(range(20))
    assert stream["c"] == expected_c


def test_row_map_preserves_declared_schema_on_empty_input():
    """A per-row `map`/`flat_map` that adds a column must carry its declared
    `output_columns` on an empty input batch, so a downstream reference to the added
    column succeeds on empty input just as it does on non-empty. Previously the empty
    fallback used the *input* schema and the downstream `select("y")` raised."""
    empty = bt.from_pydict({"x": [1, 2, 3]}).filter(col("x") > 100)

    mapped = empty.map(lambda r: {"x": r["x"], "y": r["x"] * 10}, output_columns=["x", "y"])
    assert mapped.select("y").collect().to_pydict() == {"y": []}

    flat = empty.ml.flat_map(
        lambda r: [{"x": r["x"], "y": r["x"]}, {"x": r["x"], "y": r["x"]}],
        output_columns=["x", "y"],
    )
    assert flat.select("y").collect().to_pydict() == {"y": []}

    # Non-empty path unchanged.
    full = bt.from_pydict({"x": [1, 2, 3]})
    got = full.map(lambda r: {"x": r["x"], "y": r["x"] * 10}, output_columns=["x", "y"])
    assert got.select("y").collect().to_pydict() == {"y": [10, 20, 30]}
