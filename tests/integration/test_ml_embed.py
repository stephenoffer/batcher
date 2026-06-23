"""`embed` — text column → embedding column, model loaded once per worker."""

from __future__ import annotations

import threading

import pyarrow as pa

from batcher.ml import embed


def test_embed_appends_embedding_column():
    builds = []
    lock = threading.Lock()

    def make_encoder():
        with lock:
            builds.append(1)  # one "model load" per worker
        # Deterministic fake embedding: [len(text), len(text)*2].
        return lambda texts: [[float(len(t)), float(len(t)) * 2] for t in texts]

    batches = [
        pa.RecordBatch.from_arrays([pa.array(["a", "bb"])], names=["text"]),
        pa.RecordBatch.from_arrays([pa.array(["ccc"])], names=["text"]),
    ]
    out = list(embed(batches, make_encoder, text_column="text", num_workers=2))

    # Embedding column present, original column preserved.
    flat = pa.Table.from_batches(out)
    assert flat.column_names == ["text", "embedding"]
    assert flat.column("text").to_pylist() == ["a", "bb", "ccc"]
    assert flat.column("embedding").to_pylist() == [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]
    # Model built at most once per worker (≤ num_workers), never per batch.
    assert sum(builds) <= 2


def test_embed_custom_output_column():
    out = list(
        embed(
            [pa.RecordBatch.from_arrays([pa.array(["x"])], names=["t"])],
            lambda: lambda texts: [[1.0] for _ in texts],
            text_column="t",
            output_column="vec",
            num_workers=1,
        )
    )
    assert pa.Table.from_batches(out).column_names == ["t", "vec"]
