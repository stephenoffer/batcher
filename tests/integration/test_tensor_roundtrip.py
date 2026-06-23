"""Arrow tensor columns survive the engine round-trip (D1/D2).

The canonical Arrow ``FixedShapeTensor`` extension type — the natural column for
embeddings and decoded image/audio frames — crosses the PyO3 / Arrow C Data
Interface boundary and flows through the ML-critical carry-through operators
(scan, filter, select, sort, limit, union) with its shape metadata intact, and
converts to a correctly-shaped NumPy tensor. This locks that contract in so a
future operator change can't silently downgrade tensors to their storage type.

A bare-column passthrough through `Project` carries the source field (and its
extension metadata) through, so `select`, `with_columns`, and a downstream
`distinct` all preserve tensor columns too. Aggregate/join *outputs* (genuinely new
columns) are still plain — that's correct, they aren't passthroughs.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

pytest.importorskip("batcher._native", reason="native engine not built")

import batcher as bt
from batcher import col

_TENSOR = "extension<arrow.fixed_shape_tensor"


def _tensors(n: int) -> pa.Array:
    arr = np.arange(n * 4, dtype=np.float32).reshape(n, 2, 2)
    return pa.FixedShapeTensorArray.from_numpy_ndarray(arr)


def _table(n: int = 4) -> pa.Table:
    return pa.table({"id": list(range(n)), "g": [i % 2 for i in range(n)], "t": _tensors(n)})


def _ttype(table: pa.Table) -> str:
    return str(table.schema.field("t").type)


def test_scan_preserves_tensor_type():
    assert _ttype(bt.from_arrow(_table()).collect()).startswith(_TENSOR)


def test_filter_preserves_tensor_type():
    out = bt.from_arrow(_table()).filter(col("id") > 1).collect()
    assert out.num_rows == 2
    assert _ttype(out).startswith(_TENSOR)


def test_select_passthrough_preserves_tensor_type():
    assert _ttype(bt.from_arrow(_table()).select("id", "t").collect()).startswith(_TENSOR)


def test_sort_preserves_tensor_type():
    assert _ttype(bt.from_arrow(_table()).sort("id", descending=True).collect()).startswith(_TENSOR)


def test_limit_preserves_tensor_type():
    assert _ttype(bt.from_arrow(_table()).head(2).collect()).startswith(_TENSOR)


def test_union_preserves_tensor_type():
    out = bt.from_arrow(_table(3)).union(bt.from_arrow(_table(2))).collect()
    assert out.num_rows == 5
    assert _ttype(out).startswith(_TENSOR)


def test_tensor_column_converts_to_shaped_numpy():
    batch = next(iter(bt.from_arrow(_table()).iter_batches()))
    nd = batch.column("t").to_numpy_ndarray()
    assert nd.shape == (4, 2, 2)
    assert nd.dtype == np.float32


def test_with_columns_passthrough_preserves_tensor_type():
    # The tensor column passes through unchanged alongside a new computed column;
    # the passthrough carries its extension metadata, so it stays a tensor.
    out = bt.from_arrow(_table()).with_columns(id2=col("id") + 1).collect()
    assert _ttype(out).startswith(_TENSOR)
    assert out.column("id2").to_pylist() == [1, 2, 3, 4]


def test_distinct_after_select_preserves_tensor_type():
    out = bt.from_arrow(_table()).select("t").distinct().collect()
    assert _ttype(out).startswith(_TENSOR)
