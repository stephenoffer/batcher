"""`list.slice` must not overflow on a huge offset/length (cross-area hardening).

`Expr::ListSlice` computed `begin + length` in i64 before clamping to the list end, so a
length near `i64::MAX` overflowed — a debug panic, and in release a wraparound to a giant
`usize` that aborted with "capacity overflow". Saturating arithmetic clamps to the list
end instead. DuckDB's `list_slice` clamps an over-long length the same way.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.differential


def test_list_slice_huge_length_clamps_not_panics():
    t = pa.table({"xs": pa.array([[10, 20, 30, 40], [1, 2], []], pa.list_(pa.int64()))})
    # offset 2 (0-based), a length of i64::MAX must clamp to the end, not overflow.
    out = bt.from_arrow(t).select(r=col("xs").list.slice(2, 2**63 - 1)).collect().to_pydict()
    assert out["r"] == [[30, 40], [], []]


def test_list_slice_huge_offset_is_empty():
    t = pa.table({"xs": pa.array([[10, 20, 30]], pa.list_(pa.int64()))})
    out = bt.from_arrow(t).select(r=col("xs").list.slice(2**63 - 1, 5)).collect().to_pydict()
    assert out["r"] == [[]]
