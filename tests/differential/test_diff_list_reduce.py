"""Per-row list reductions (product/std/var) + unique — structural."""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col


def _ints():
    return bt.from_arrow(
        pa.table({"a": pa.array([[3, 1, 2], [4], [], None], type=pa.list_(pa.int64()))})
    )


def test_list_product():
    out = _ints().select(p=col("a").list.product()).collect().to_pydict()
    assert out["p"] == [6.0, 4.0, None, None]


def test_list_var():
    # [3,1,2]: mean=2 → var=((3-2)²+(1-2)²+(2-2)²)/(3-1)=(1+1+0)/2=1.0
    # [4]: n<2 → null; []: empty → null; None → null.
    out = _ints().select(v=col("a").list.var()).collect().to_pydict()
    assert out["v"] == [1.0, None, None, None]


def test_list_std():
    # std = sqrt(var); for [3,1,2] var=1.0 → std=1.0.
    out = _ints().select(s=col("a").list.std()).collect().to_pydict()
    assert out["s"] == [1.0, None, None, None]


def test_list_unique():
    out = _ints().select(u=col("a").list.unique()).collect().to_pydict()
    assert out["u"] == [[3, 1, 2], [4], [], None]


def test_list_unique_dedups_first_occurrence():
    ds = bt.from_arrow(pa.table({"a": pa.array([[1, 1, 2, 2, 3]], type=pa.list_(pa.int64()))}))
    out = ds.select(u=col("a").list.unique()).collect().to_pydict()
    assert out["u"] == [[1, 2, 3]]


def test_list_var_known_spread():
    # [2,4,4,4,5,5,7,9]: mean=5, Σ(x-mean)²=32, sample var=32/7.
    ds = bt.from_arrow(
        pa.table({"a": pa.array([[2, 4, 4, 4, 5, 5, 7, 9]], type=pa.list_(pa.int64()))})
    )
    out = ds.select(v=col("a").list.var(), s=col("a").list.std()).collect().to_pydict()
    assert out["v"][0] == 32.0 / 7.0
    assert out["s"][0] == (32.0 / 7.0) ** 0.5
