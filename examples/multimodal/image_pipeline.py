"""An end-to-end multimodal pipeline: read, filter, decode, score, write.

The ordering is the whole lesson. Metadata filter first, because it is free; decode second,
because it is expensive; score third, on whatever survived. Reversing any two of those
multiplies the cost of the pipeline by the selectivity of the filter you skipped.

    python examples/multimodal/image_pipeline.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow as pa

import batcher as bt
from _common import images
from batcher import col


def main() -> None:
    # 1. Read headers only.
    catalog = bt.read.images(images(100))
    print("images found:", catalog.count())
    assert catalog.count() == 100

    # 2. Filter on metadata — no decode has happened yet.
    wanted = catalog.filter((col("size") > 3_000) & (col("width") > 0))
    print("after the metadata filter:", wanted.count())
    assert wanted.count() <= catalog.count()

    # 3. Decode only what survived, at the size the model wants.
    uris = wanted.select("uri").to_pydict()["uri"]
    assert uris
    decoded = bt.read.images(images(100), decode=True, size=(32, 32)).filter(col("uri").is_in(uris))
    assert decoded.count() == wanted.count()

    # 4. Score, batch at a time.
    def brightness(batch: pa.RecordBatch) -> pa.RecordBatch:
        tensors = batch.column("image").to_pylist()
        scores = [sum(values) / len(values) if values else 0.0 for values in tensors]
        return pa.RecordBatch.from_arrays(
            [*batch.columns, pa.array(scores, type=pa.float64())],
            names=[*batch.schema.names, "brightness"],
        )

    scored = decoded.select("uri", "image").map_batches(brightness)
    # A Python callback's output schema is not known until it runs, so the new column
    # cannot be projected in the same plan — materialize, then carry on.
    materialized = scored.to_pydict()
    result = {"uri": materialized["uri"], "brightness": materialized["brightness"]}
    print("brightest:", round(max(result["brightness"]), 2))

    assert len(result["brightness"]) == decoded.count()
    assert all(value >= 0 for value in result["brightness"])

    # 5. Write the scores, dropping the pixels — the bytes were never the output.
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "scores.parquet")
        bt.from_pydict(result).write.parquet(path)
        back = bt.read.parquet(path)
        assert back.count() == decoded.count()
        assert set(back.columns) == {"uri", "brightness"}
        print("wrote", back.count(), "scores")


if __name__ == "__main__":
    main()
