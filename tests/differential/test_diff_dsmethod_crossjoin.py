"""``Dataset.cross_join`` vs DuckDB — including the reserved-key-name collision.

``cross_join`` lowers to an equi-join on a synthetic constant key column that it adds
with ``with_columns`` and drops afterwards. Because ``with_columns`` *replaces* a
same-named column, a user column that happens to share the synthetic key's name would be
silently overwritten by the constant and then dropped with it — losing the column and its
data with no error. The join must instead pick a key name absent from both inputs, so a
column named exactly ``__cross_key__`` survives the Cartesian product unchanged.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential


def test_cross_join_basic(duck):
    left = pa.table({"a": [1, 2, 3], "v": [10, 20, 30]})
    right = pa.table({"b": ["x", "y"]})
    got = bt.from_arrow(left).cross_join(bt.from_arrow(right)).collect()
    duck.register("l", left)
    duck.register("r", right)
    assert_same(got, duck.sql("SELECT * FROM l CROSS JOIN r"))


def test_cross_join_preserves_reserved_key_column(duck):
    """A right column literally named ``__cross_key__`` must not be dropped/overwritten."""
    left = pa.table({"a": [1, 2], "v": [10, 20]})
    right = pa.table({"__cross_key__": [7, 8]})
    got = bt.from_arrow(left).cross_join(bt.from_arrow(right)).collect()
    assert "__cross_key__" in got.column_names
    duck.register("l", left)
    duck.register("r", right)
    assert_same(got, duck.sql("SELECT * FROM l CROSS JOIN r"))


def test_cross_join_reserved_key_on_left(duck):
    left = pa.table({"__cross_key__": [1, 2], "z": [9, 9]})
    right = pa.table({"b": ["x"]})
    got = bt.from_arrow(left).cross_join(bt.from_arrow(right)).collect()
    assert "__cross_key__" in got.column_names
    duck.register("l", left)
    duck.register("r", right)
    assert_same(got, duck.sql("SELECT * FROM l CROSS JOIN r"))
