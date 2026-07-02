"""`map_batches(batch_format=...)` — the UDF speaks numpy / pandas / torch / arrow.

Each format converts only around the per-batch call; the engine boundary stays
Arrow. An identity UDF in every format must reproduce the input exactly, and a
compute UDF must agree across formats.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError


def _table() -> pa.Table:
    return pa.table({"x": [1, 2, 3, 4], "y": [10, 20, 30, 40]})


def test_pyarrow_default_identity():
    out = bt.from_arrow(_table()).map_batches(lambda b: b).collect()
    assert out.to_pydict() == _table().to_pydict()


def test_numpy_format_compute():
    def add(d: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {"x": d["x"], "z": d["x"] + d["y"]}

    out = bt.from_arrow(_table()).map_batches(add, batch_format="numpy").collect()
    assert out.column("z").to_pylist() == [11, 22, 33, 44]


def test_pandas_format_compute():
    pytest.importorskip("pandas")

    def add(df):
        df["z"] = df["x"] * df["y"]
        return df

    out = bt.from_arrow(_table()).map_batches(add, batch_format="pandas").collect()
    assert out.column("z").to_pylist() == [10, 40, 90, 160]


def test_torch_format_compute():
    pytest.importorskip("torch")

    def add(d):
        return {"x": d["x"], "z": d["x"] + d["y"]}

    out = (
        bt.from_arrow(_table())
        .map_batches(add, batch_format="torch", output_columns=["x", "z"])
        .collect()
    )
    assert out.column("z").to_pylist() == [11, 22, 33, 44]


def test_formats_agree():
    pytest.importorskip("pandas")
    pytest.importorskip("torch")
    t = pa.table({"x": list(range(50)), "y": list(range(100, 150))})

    def np_add(d):
        return {"r": d["x"] + d["y"]}

    def pd_add(df):
        return df.assign(r=df["x"] + df["y"])[["r"]]

    def t_add(d):
        return {"r": d["x"] + d["y"]}

    rn = bt.from_arrow(t).map_batches(np_add, batch_format="numpy").collect()
    rp = bt.from_arrow(t).map_batches(pd_add, batch_format="pandas").collect()
    rt = bt.from_arrow(t).map_batches(t_add, batch_format="torch").collect()
    assert rn.column("r").to_pylist() == rp.column("r").to_pylist()
    assert rn.column("r").to_pylist() == rt.column("r").to_pylist()


def test_unknown_format_rejected():
    with pytest.raises(PlanError, match="batch_format"):
        bt.from_arrow(_table()).map_batches(lambda b: b, batch_format="polars")


# --------------------------------------------------------------------------- #
# Multi-dimensional tensor columns (images / embeddings / feature maps)
# --------------------------------------------------------------------------- #
def test_map_batches_emits_multidim_tensor_column():
    """A `fn` returning a ``(B, *shape)`` NumPy array yields a canonical Arrow
    fixed-shape-tensor column (the Ray Data tensor-block shape) — not an error."""

    def make(b: pa.RecordBatch) -> dict:
        n = b.num_rows
        ids = b.column("x").to_numpy()
        img = (ids[:, None, None, None] + np.zeros((n, 3, 4, 4), np.float32)).astype(np.float32)
        return {"x": ids, "img": img}

    out = (
        bt.from_arrow(_table())
        .map_batches(make, output_columns=["x", "img"], batch_format="pyarrow")
        .collect()
    )
    from batcher.io.formats.ml.tensor import is_tensor_column

    assert out.num_rows == 4
    field = out.schema.field("img")
    assert is_tensor_column(field.type)
    assert tuple(field.type.shape) == (3, 4, 4)


def test_tensor_column_round_trips_through_numpy_stage():
    """A tensor column produced by one stage is read back as a ``(B, *shape)`` ndarray
    by a downstream ``batch_format="numpy"`` stage — the two-stage decode→model shape."""

    def make(b: pa.RecordBatch) -> dict:
        n = b.num_rows
        ids = b.column("x").to_numpy().astype(np.float32)
        return {"x": b.column("x").to_numpy(), "t": ids[:, None] + np.zeros((n, 6), np.float32)}

    def reduce(d: dict) -> dict:
        assert d["t"].shape[1:] == (6,)
        return {"x": d["x"], "s": d["t"].sum(1).astype(np.float64)}

    out = (
        bt.from_arrow(_table())
        .map_batches(make, output_columns=["x", "t"], batch_format="pyarrow")
        .map_batches(reduce, output_columns=["x", "s"], batch_format="numpy")
        .collect()
    )
    got = dict(zip(out.to_pydict()["x"], out.to_pydict()["s"], strict=False))
    assert got == {1: 6.0, 2: 12.0, 3: 18.0, 4: 24.0}  # each id filled 6 cells


# --------------------------------------------------------------------------- #
# Dirty-data tolerance (max_errored_rows)
# --------------------------------------------------------------------------- #
def _flaky(d: dict) -> dict:
    """A numpy-format UDF that raises on any row whose x is divisible by 7."""
    x = d["x"]
    if (x % 7 == 0).any():
        raise ValueError("corrupt row")
    return {"x": x, "y": (x * 2).astype(np.int64)}


def test_max_errored_rows_strict_default_raises():
    t = pa.table({"x": np.arange(1, 50, dtype=np.int64)})  # contains multiples of 7
    with pytest.raises(Exception, match="corrupt"):
        bt.from_arrow(t).map_batches(
            _flaky, output_columns=["x", "y"], batch_format="numpy"
        ).collect()


def test_max_errored_rows_skips_bad_rows():
    t = pa.table({"x": np.arange(1, 200, dtype=np.int64)})
    bad = [v for v in range(1, 200) if v % 7 == 0]
    out = (
        bt.from_arrow(t)
        .map_batches(_flaky, output_columns=["x", "y"], batch_format="numpy", max_errored_rows=50)
        .collect()
    )
    xs = out.to_pydict()["x"]
    assert out.num_rows == 199 - len(bad)
    assert not any(v % 7 == 0 for v in xs)  # corrupt rows dropped
    assert out.to_pydict()["y"][:3] == [2, 4, 6]  # surviving rows computed correctly


def test_max_errored_rows_budget_exhausted_raises():
    t = pa.table({"x": np.arange(1, 200, dtype=np.int64)})  # ~28 bad rows
    with pytest.raises(Exception, match="corrupt"):
        bt.from_arrow(t).map_batches(
            _flaky, output_columns=["x", "y"], batch_format="numpy", max_errored_rows=5
        ).collect()
