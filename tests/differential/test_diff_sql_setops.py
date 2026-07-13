"""Set-operation differential tests vs DuckDB — the DISTINCT and the ALL forms.

The ALL forms carry multiplicity (``INTERSECT ALL`` keeps a row min(l, r) times,
``EXCEPT ALL`` keeps it max(l - r, 0) times), which a plain membership test cannot
express. These pin that multiplicity — and the NULL-compares-equal set semantics —
against the oracle.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

# (left rows, right rows) — duplicates, disjoint, nulls, and a right side that
# subtracts more copies than the left has.
_DATA = [
    ([1, 1, 2, 3], [1, 3]),
    ([1, 1, 2], [1, 1, 3]),
    ([1, None, None, 2], [None, 2]),
    ([5, 5, 5], [5]),
    ([1, 2], [3, 4]),
    ([1], [1, 1, 1]),
]

_OPS = ["UNION", "UNION ALL", "INTERSECT", "INTERSECT ALL", "EXCEPT", "EXCEPT ALL"]


@pytest.mark.differential
@pytest.mark.parametrize("op", _OPS)
@pytest.mark.parametrize("left,right", _DATA)
def test_diff_setop(duck, op, left, right):
    from conftest import assert_same

    lt, rt = pa.table({"x": left}), pa.table({"x": right})
    duck.register("l", lt)
    duck.register("r", rt)
    query = f"SELECT x FROM l {op} SELECT x FROM r"
    got = bt.sql(query, l=bt.from_arrow(lt), r=bt.from_arrow(rt)).collect()
    assert_same(got, duck.sql(query))


@pytest.mark.differential
@pytest.mark.parametrize("op", _OPS)
def test_diff_setop_multi_column(duck, op):
    """Multi-column set ops: the whole row is the key, nulls included."""
    from conftest import assert_same

    lt = pa.table({"a": [1, 1, 2, None], "b": ["x", "x", "y", None]})
    rt = pa.table({"a": [1, None], "b": ["x", None]})
    duck.register("l", lt)
    duck.register("r", rt)
    query = f"SELECT a, b FROM l {op} SELECT a, b FROM r"
    got = bt.sql(query, l=bt.from_arrow(lt), r=bt.from_arrow(rt)).collect()
    assert_same(got, duck.sql(query))
