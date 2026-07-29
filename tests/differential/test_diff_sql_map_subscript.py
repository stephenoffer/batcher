"""Map lookup and `map_keys` from SQL — vs DuckDB.

Three spellings of one operation, none of which reached the kernel that already
implements it:

* `m['a']` raised `invalid literal for int()`. The `exp.Bracket` handler assumed every
  subscript was a list index and called `int()` on the key, so an ordinary map lookup
  crashed with an error about integers.
* Spark's `element_at(m, 'a')` parses as the same node, so it crashed the same way.
* `map_keys(m)` reported "unsupported SQL expression" while `map_values(m)` worked — the
  two differ only in that sqlglot gives one a typed node and leaves the other anonymous.

The fixture separates the cases a map lookup gets wrong: a **present** key, an **absent**
key in a non-empty map, an **empty** map, and a **null** map. A null map answers NULL and
an absent key answers NULL, but they arrive by different routes, and a kernel that
conflated them would still pass a present-key-only test.

`map_extract` is *not* here, deliberately. DuckDB returns a one-element list for a hit and
an empty list for a miss, where the subscript returns the bare value; see the comment on
`_MAP_KEY` in `_sql/parser/expressions/anonymous.py`.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_ordered


@pytest.fixture
def maps(duck):
    t = pa.table(
        {
            "k": [0, 1, 2, 3],
            "m": pa.array(
                [[("a", 1), ("b", 2)], [("c", 3)], None, []],
                type=pa.map_(pa.string(), pa.int64()),
            ),
            "l": [[10, 20], [30], [40], [50]],
        }
    )
    duck.register("t", t)
    return t


@pytest.mark.differential
@pytest.mark.parametrize(
    "query",
    [
        "SELECT k, m['a'] r FROM t ORDER BY k",
        "SELECT k, m['zz'] r FROM t ORDER BY k",
        "SELECT k, map_keys(m) r FROM t ORDER BY k",
        "SELECT k, map_values(m) r FROM t ORDER BY k",
    ],
)
def test_map_access_matches_duckdb(duck, maps, query):
    assert_same_ordered(bt.sql(query, t=maps).collect(), duck.sql(query))


@pytest.mark.differential
def test_a_string_subscript_does_not_crash_on_the_integer_path(maps):
    """The regression this file exists for: the error was about integers, on a query with
    no integer in it."""
    rows = bt.sql("SELECT k, m['a'] r FROM t ORDER BY k", t=maps).to_pydict()
    assert rows["r"] == [1, None, None, None]


@pytest.mark.differential
def test_spark_element_at_on_a_map_reaches_the_map_kernel(maps):
    """Spark's `element_at(m, key)` parses as the same `Bracket` node as `m[key]`, so it
    took the list path and raised too."""
    rows = bt.sql("SELECT k, element_at(m, 'a') r FROM t ORDER BY k", t=maps, dialect="spark")
    assert rows.to_pydict()["r"] == [1, None, None, None]


@pytest.mark.differential
def test_an_integer_subscript_is_still_a_list_index(duck, maps):
    """The load-bearing negative test. Dispatching on the key's type must not move a list
    subscript onto the map path — and a list index stays 1-based in DuckDB."""
    query = "SELECT k, l[1] r FROM t ORDER BY k"
    assert_same_ordered(bt.sql(query, t=maps).collect(), duck.sql(query))
    assert bt.sql(query, t=maps).to_pydict()["r"] == [10, 30, 40, 50]


@pytest.mark.differential
def test_spark_list_element_at_keeps_its_one_based_offset(maps):
    """Spark records the 1-based origin in `Bracket.offset`; the map branch is taken before
    that is read, so this pins that integer keys still go through it."""
    rows = bt.sql("SELECT k, element_at(l, 2) r FROM t ORDER BY k", t=maps, dialect="spark")
    assert rows.to_pydict()["r"] == [20, None, None, None]
