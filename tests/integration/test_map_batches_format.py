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
