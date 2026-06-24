"""`Dataset.with_row_index` (Polars parity) — a sequential row-index column.

Not a DuckDB-differential op (it follows Polars semantics). The correctness points:
contiguous numbering across batches, and that a filter ABOVE `with_row_index` keeps
the original numbering (the predicate must not push below the RowId barrier), while a
filter BELOW it renumbers.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.integration


def test_basic_and_offset():
    out = bt.from_pydict({"x": ["a", "b", "c"]}).with_row_index().to_pydict()
    assert out == {"index": [0, 1, 2], "x": ["a", "b", "c"]}
    out2 = bt.from_pydict({"x": [10, 20]}).with_row_index("id", offset=100).to_pydict()
    assert out2 == {"id": [100, 101], "x": [10, 20]}


def test_contiguous_across_batches():
    # Many rows → multiple morsels; ids stay a contiguous 0..n-1 run.
    big = bt.from_arrow(pa.table({"v": list(range(5000))}))
    idx = big.with_row_index().collect().to_pydict()["index"]
    assert idx == list(range(5000))


def test_filter_below_renumbers_but_above_preserves():
    ds = bt.from_pydict({"x": [1, 2, 3, 4, 5]})
    # Filter then number → fresh 0-based numbering over the kept rows.
    below = ds.filter(col("x") > 2).with_row_index().to_pydict()
    assert below == {"index": [0, 1, 2], "x": [3, 4, 5]}
    # Number then filter → the original row numbers survive (no push-through).
    above = ds.with_row_index().filter(col("x") > 2).to_pydict()
    assert above == {"index": [2, 3, 4], "x": [3, 4, 5]}


def test_projection_prunes_input_but_keeps_index():
    big = bt.from_arrow(pa.table({"v": list(range(10))}))
    # Selecting only the index must still produce the full 0..9 sequence.
    idx = big.with_row_index().select("index").collect().to_pydict()["index"]
    assert idx == list(range(10))


def test_name_collision_raises():
    from batcher._internal.errors import PlanError

    with pytest.raises(PlanError, match="collides"):
        bt.from_pydict({"index": [1, 2]}).with_row_index()
