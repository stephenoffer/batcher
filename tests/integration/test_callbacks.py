"""Integration coverage for the Ray-Data-style callback transforms.

`map`/`flat_map`/`@udf` are black-box Python callbacks (no DuckDB oracle), routed
through the worker-side `map_batches` path. The per-row function runs in the worker
(data plane), never the driver.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher import udf

pytestmark = pytest.mark.integration


def _ds():
    return bt.from_pydict({"a": [1, 2, 3], "b": [10, 20, 30]})


def test_map_per_row_adds_column():
    out = (
        _ds().map(lambda r: {"a": r["a"], "b": r["b"], "c": r["a"] + r["b"]}).collect().to_pydict()
    )
    assert out == {"a": [1, 2, 3], "b": [10, 20, 30], "c": [11, 22, 33]}


def test_flat_map_one_to_many():
    out = (
        _ds()
        .flat_map(lambda r: [{"a": r["a"], "k": i} for i in range(r["a"])])
        .collect()
        .to_pydict()
    )
    assert out == {"a": [1, 2, 2, 3, 3, 3], "k": [0, 0, 1, 0, 1, 2]}


def test_flat_map_can_drop_rows():
    # Returning [] for a row drops it (a filtering flat_map).
    out = _ds().flat_map(lambda r: [r] if r["a"] % 2 else []).collect().to_pydict()
    assert out == {"a": [1, 3], "b": [10, 30]}


def test_udf_per_row_and_batch():
    @udf(per_row=True)
    def doubled(r):
        return {"a": r["a"], "d": r["a"] * 2}

    assert doubled(_ds()).collect().to_pydict() == {"a": [1, 2, 3], "d": [2, 4, 6]}

    @udf()
    def scale(batch):
        import pyarrow as pa

        return batch.append_column("x", pa.array([v.as_py() * 100 for v in batch.column("a")]))

    assert scale(_ds()).collect().to_pydict()["x"] == [100, 200, 300]
