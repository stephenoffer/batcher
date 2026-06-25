"""List/array reduction (`.list`) tests — structural (hand-computed expected).

DuckDB's list-aggregate function names vary by version, so these assert against
explicit expected values that encode Batcher's semantics: null list rows stay
null; empty lists reduce to null for sum/min/max/mean but 0 for len/n_unique.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col


def _lists():
    return bt.from_arrow(
        pa.table(
            {
                "a": pa.array(
                    [[1, 2, 3], [4, 5], [], None, [7, 7, 7]],
                    type=pa.list_(pa.int64()),
                ),
            }
        )
    )


def test_list_len():
    out = _lists().select(n=col("a").list.len()).collect().to_pydict()
    assert out["n"] == [3, 2, 0, None, 3]


def test_list_sum_min_max_mean():
    out = (
        _lists()
        .select(
            s=col("a").list.sum(),
            mn=col("a").list.min(),
            mx=col("a").list.max(),
            av=col("a").list.mean(),
        )
        .collect()
        .to_pydict()
    )
    assert out["s"] == [6.0, 9.0, None, None, 21.0]
    assert out["mn"] == [1.0, 4.0, None, None, 7.0]
    assert out["mx"] == [3.0, 5.0, None, None, 7.0]
    assert out["av"] == [2.0, 4.5, None, None, 7.0]


def test_list_n_unique():
    out = _lists().select(u=col("a").list.n_unique()).collect().to_pydict()
    assert out["u"] == [3, 2, 0, None, 1]


def test_list_float():
    ds = bt.from_arrow(pa.table({"a": pa.array([[1.5, 2.5], [10.0]], type=pa.list_(pa.float64()))}))
    out = ds.select(s=col("a").list.sum(), n=col("a").list.len()).collect().to_pydict()
    assert out["s"] == [4.0, 10.0]
    assert out["n"] == [2, 1]
