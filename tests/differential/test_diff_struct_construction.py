"""Differential coverage for struct construction (`struct` / `named_struct`).

Both build a `MakeStruct` node; reading a field back via `.struct.field` round-trips.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, named_struct, struct

pytestmark = pytest.mark.differential


def _data():
    return pa.table({"a": [1, 2, 3], "b": [10, 20, 30]})


def test_struct_pack_matches_duckdb(duck):
    # The order-independent harness can't sort struct (dict) columns, so compare
    # pydicts directly — a plain scan+project preserves row order on both sides.
    duck.register("t", _data())
    out = bt.from_arrow(_data()).select(s=struct(x=col("a"), y=col("b"))).collect().to_pydict()
    expected = duck.sql("SELECT {'x': a, 'y': b} AS s FROM t").to_arrow_table().to_pydict()
    assert out == expected


def test_named_struct_equals_struct():
    ds = bt.from_arrow(_data())
    a = ds.select(s=struct(p=col("a"), q=col("b"))).collect().to_pydict()
    b = ds.select(s=named_struct("p", col("a"), "q", col("b"))).collect().to_pydict()
    assert a == b


def test_struct_field_roundtrip():
    ds = bt.from_arrow(_data())
    out = (
        ds.select(s=struct(x=col("a"), y=col("b") + 1))
        .select(x=col("s").struct.field("x"), y=col("s").struct.field("y"))
        .collect()
        .to_pydict()
    )
    assert out == {"x": [1, 2, 3], "y": [11, 21, 31]}
