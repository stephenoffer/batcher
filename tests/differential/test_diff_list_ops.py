"""List-returning ops (sort/reverse/slice) + list.contains — structural."""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col


def _ints():
    return bt.from_arrow(
        pa.table({"a": pa.array([[3, 1, 2], [5], [], None], type=pa.list_(pa.int64()))})
    )


def test_list_reverse():
    out = _ints().select(r=col("a").list.reverse()).collect().to_pydict()
    assert out["r"] == [[2, 1, 3], [5], [], None]


def test_list_sort():
    out = _ints().select(s=col("a").list.sort()).collect().to_pydict()
    assert out["s"] == [[1, 2, 3], [5], [], None]


def test_list_slice():
    out = (
        _ints().select(s=col("a").list.slice(1, 2), s0=col("a").list.slice(1)).collect().to_pydict()
    )
    assert out["s"] == [[1, 2], [], [], None]  # offset 1, length 2
    assert out["s0"] == [[1, 2], [], [], None]  # offset 1, to end


def test_list_contains_int():
    out = (
        _ints()
        .select(has2=col("a").list.contains(2), has9=col("a").list.contains(9))
        .collect()
        .to_pydict()
    )
    assert out["has2"] == [True, False, False, None]
    assert out["has9"] == [False, False, False, None]


def test_list_contains_string():
    ds = bt.from_arrow(
        pa.table({"a": pa.array([["x", "y"], ["z"], []], type=pa.list_(pa.string()))})
    )
    out = ds.select(hx=col("a").list.contains("x")).collect().to_pydict()
    assert out["hx"] == [True, False, False]


def test_sort_then_get():
    # Compose list ops: smallest element via sort + get(0).
    out = _ints().select(mn=col("a").list.sort().list.get(0)).collect().to_pydict()
    assert out["mn"] == [1, 5, None, None]
