"""List-returning ops (sort/reverse/slice) and `list.contains`, against DuckDB.

This file was marked "structural" and asserted only hand-written values, which put it in
`tests/differential/` with no differential in it. DuckDB has `list_reverse`, `list_sort`,
`list_contains` and list slicing, so the oracle was available and simply absent —
`.claude/rules/testing.md` requires it for any expression behaviour.

Checked, the two engines agree on every case here, nulls and empty lists included. The
explicit expected values are kept beside the oracle comparison rather than replaced by it:
they document what the semantics *are*, which a cross-engine assertion alone does not, and
they would catch the two engines drifting together (a shared pyarrow kernel changing under
both, say).

**One mapping is worth stating, because it is the one place a reader could be misled.**
Batcher's `list.slice(offset, length)` is 0-based with a length; DuckDB's `a[from:to]` is
1-based and inclusive on both ends. So `list.slice(1, 2)` is `a[2:3]`, not `a[1:2]`. Writing
the obvious-looking `a[1:2]` here would have produced a passing-looking oracle that compared
a different operation.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col

pytestmark = pytest.mark.differential

pytest.importorskip("duckdb")

#: The one input every case runs on: an unsorted list, a singleton, an empty list, and a
#: null. The last two are where list kernels most often disagree — an empty list is not a
#: null list, and an implementation that conflates them passes anything built on the first
#: two rows.
_INTS = pa.table({"a": pa.array([[3, 1, 2], [5], [], None], type=pa.list_(pa.int64()))})


def _duck(duck, sql: str):
    """Register `_INTS` as `t` and return the relation for `sql`."""
    duck.register("t", _INTS)
    return duck.sql(sql)


def _ints():
    return bt.from_arrow(_INTS)


def test_list_reverse(duck):
    got = _ints().select(r=col("a").list.reverse()).collect()
    assert got.to_pydict()["r"] == [[2, 1, 3], [5], [], None]
    assert_same(got, _duck(duck, "SELECT list_reverse(a) AS r FROM t"))


def test_list_sort(duck):
    got = _ints().select(s=col("a").list.sort()).collect()
    assert got.to_pydict()["s"] == [[1, 2, 3], [5], [], None]
    assert_same(got, _duck(duck, "SELECT list_sort(a) AS s FROM t"))


def test_list_slice(duck):
    out = (
        _ints().select(s=col("a").list.slice(1, 2), s0=col("a").list.slice(1)).collect().to_pydict()
    )
    assert out["s"] == [[1, 2], [], [], None]  # offset 1, length 2
    assert out["s0"] == [[1, 2], [], [], None]  # offset 1, to end
    # `a[2:3]`, not `a[1:2]` — see the module docstring on the index mapping.
    assert_same(
        _ints().select(s=col("a").list.slice(1, 2)).collect(),
        _duck(duck, "SELECT a[2:3] AS s FROM t"),
    )


def test_list_contains_int(duck):
    got = _ints().select(has2=col("a").list.contains(2), has9=col("a").list.contains(9)).collect()
    out = got.to_pydict()
    assert out["has2"] == [True, False, False, None]
    assert out["has9"] == [False, False, False, None]
    # A null list yields a null answer, not False — the distinction an `IS NOT NULL` filter
    # downstream depends on, and the one both engines have to agree about.
    assert_same(
        got,
        _duck(
            duck,
            "SELECT list_contains(a, 2) AS has2, list_contains(a, 9) AS has9 FROM t",
        ),
    )


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
