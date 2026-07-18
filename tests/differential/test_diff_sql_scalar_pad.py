"""SQL scalar-function edge-case regression tests vs DuckDB (scalar hunt).

Each case pins a defect in the `_sql` translator's scalar-function path. The
`lpad`/`rpad` cases crashed on a negative pad width (a `Neg`-wrapped literal that
`int(node.this)` could not consume) before the fix; DuckDB clamps a non-positive
width to the empty string, which the engine's `.lpad`/`.rpad` already do.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from conftest import assert_same


@pytest.fixture
def tbl(duck):
    t = pa.table({"s": pa.array(["hi", "hello", "", "abc"], pa.string())})
    duck.register("t", t)
    return {"t": t}


@pytest.mark.differential
@pytest.mark.parametrize(
    "q",
    [
        # Negative width — parsed as Neg(Literal); DuckDB clamps to ''.
        "SELECT lpad(s, -1, '*') AS x FROM t",
        "SELECT rpad(s, -5, '*') AS x FROM t",
        # Zero width — also clamps to ''.
        "SELECT lpad(s, 0, '*') AS x FROM t",
        "SELECT rpad(s, 0, '*') AS x FROM t",
        # Positive widths still work (truncate / pad).
        "SELECT lpad(s, 6, '*') AS x FROM t",
        "SELECT rpad(s, 3, '*') AS x FROM t",
        "SELECT lpad(s, 7, 'xy') AS x FROM t",
    ],
)
def test_pad_width_edges(duck, tbl, q):
    got = bt.sql(q, **tbl).collect()
    assert_same(got, duck.sql(q))
