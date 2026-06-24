"""Coverage for `list.intersect`/`list.difference` (Spark array_intersect/except).

`intersect` cross-checks DuckDB `list_intersect`; `except` has no portable DuckDB
function, so its oracle is an order-preserving Python reference.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.differential


def _data():
    return pa.table(
        {
            "a": pa.array([[1, 2, 2, 3], [5, 6], [], None], type=pa.list_(pa.int64())),
            "b": pa.array([[2, 3, 4], [7], [1], [1]], type=pa.list_(pa.int64())),
        }
    )


def _distinct_order(keep):
    seen, out = set(), []
    for x in keep:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def test_intersect_matches_duckdb(duck):
    from conftest import assert_same

    duck.register("t", _data())
    out = bt.from_arrow(_data()).select(i=col("a").list.intersect(col("b"))).collect()
    assert_same(out, duck.sql("SELECT list_intersect(a, b) AS i FROM t"))


def test_difference_order_preserving_reference():
    rows = _data().to_pydict()
    out = (
        bt.from_arrow(_data())
        .select(d=col("a").list.difference(col("b")))
        .collect()
        .to_pydict()["d"]
    )
    for a, b, got in zip(rows["a"], rows["b"], out, strict=True):
        if a is None:
            assert got is None
        else:
            assert got == _distinct_order([x for x in a if x not in set(b or [])])


def test_union_order_preserving_reference():
    # DuckDB's list_distinct(list_concat(...)) does not preserve first-occurrence
    # order; Spark array_union does, so the oracle is an order-preserving reference.
    rows = _data().to_pydict()
    out = bt.from_arrow(_data()).select(u=col("a").list.union(col("b"))).collect().to_pydict()["u"]
    for a, b, got in zip(rows["a"], rows["b"], out, strict=True):
        if a is None:
            assert got is None
        else:
            assert got == _distinct_order(list(a) + list(b or []))
