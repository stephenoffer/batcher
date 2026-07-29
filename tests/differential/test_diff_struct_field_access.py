"""Struct field access, in every spelling — vs DuckDB.

The `.struct` namespace had exactly one method (`field`) against DuckDB's several, and
none of SQL's three spellings of a field access reached even that one:

* `s['a']` hit the `element_at` kernel, which rejected anything that was not a `Map`;
* `struct_extract(s, 'a')` was an unhandled typed node;
* `struct_keys(s)` had no implementation at all.

The fix is one kernel rather than three: a struct is a keyed container too, so
`element_at` resolves a *name* against the struct's fields where it scans a map's entries.
That is deliberate — the translator has no schema and cannot tell `s['a']` from `m['a']`,
so the disambiguation belongs where the array's type is actually known.

What that kernel has to get right, and what this file checks:

* **A null struct row answers null**, even though its children hold values underneath.
  Arrow keeps the null mask on the parent, so a bare `column_by_name` returns the child's
  buffer and silently resurrects a value inside a null row.
* **An absent field is an error, not a null** — the opposite of a map, whose missing key
  is an ordinary result. A struct's fields are fixed by its type, so naming one it lacks
  is a mistake.
* **`struct_keys` repeats the same list on every row** but still nulls out a null row,
  which is what stops it being a constant.

The dot form `s.a` is *not* covered: sqlglot parses it as a `Column` qualified by table
`s`, so it is rejected during column resolution, before expression dispatch. That is a
different layer and remains open.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_ordered
from batcher import col

SQL = [
    "SELECT k, s['a'] r FROM t ORDER BY k",
    "SELECT k, struct_extract(s, 'a') r FROM t ORDER BY k",
    "SELECT k, struct_keys(s) r FROM t ORDER BY k",
    "SELECT k, s['b'] r FROM t ORDER BY k",
]


@pytest.fixture
def structs(duck):
    """Row 1 is a null struct whose children still hold values in their buffers."""
    t = pa.table(
        {
            "k": [0, 1, 2],
            "s": pa.array([{"a": 1, "b": "x"}, None, {"a": 3, "b": "z"}]),
        }
    )
    duck.register("t", t)
    return t


@pytest.mark.differential
@pytest.mark.parametrize("query", SQL)
def test_sql_struct_access_matches_duckdb(duck, structs, query):
    assert_same_ordered(bt.sql(query, t=structs).collect(), duck.sql(query))


@pytest.mark.differential
def test_the_dataframe_spellings_agree_with_each_other(structs):
    """`.struct.field`, `.struct.get` and the SQL subscript are one operation."""
    ds = bt.from_arrow(structs)
    by_field = ds.select(r=col("s").struct.field("a")).to_pydict()["r"]
    by_get = ds.select(r=col("s").struct.get("a")).to_pydict()["r"]
    by_sql = bt.sql("SELECT s['a'] r FROM t ORDER BY k", t=structs).to_pydict()["r"]
    assert by_field == by_get == by_sql == [1, None, 3]


@pytest.mark.differential
def test_a_null_struct_row_stays_null(duck, structs):
    """The case a bare child lookup gets wrong: the child buffer holds a value under the
    parent's null, so returning it unmerged resurrects a row DuckDB reports as null."""
    got = bt.from_arrow(structs).select(r=col("s").struct.get("a")).to_pydict()["r"]
    expected = duck.sql("SELECT struct_extract(s,'a') r FROM t ORDER BY k")
    assert got[1] is None
    assert got == expected.arrow().read_all().to_pydict()["r"]


@pytest.mark.differential
def test_struct_keys_repeats_the_type_but_not_over_a_null_row(duck, structs):
    got = bt.from_arrow(structs).select(r=col("s").struct.keys()).to_pydict()["r"]
    assert got == [["a", "b"], None, ["a", "b"]]
    assert_same_ordered(
        bt.from_arrow(structs).select(k=col("k"), r=col("s").struct.keys()).sort("k").collect(),
        duck.sql("SELECT k, struct_keys(s) r FROM t ORDER BY k"),
    )


@pytest.mark.differential
def test_an_absent_field_is_an_error_not_a_null(structs):
    """A struct is not a map: its fields come from its type, so naming a missing one is a
    mistake rather than a lookup that found nothing."""
    with pytest.raises(Exception, match=r"nope|field"):
        bt.from_arrow(structs).select(r=col("s").struct.get("nope")).collect()


@pytest.mark.differential
def test_a_map_subscript_still_works(duck):
    """The negative test for the shared kernel: teaching `element_at` about structs must
    not disturb the map path it already served."""
    m = pa.table(
        {"k": [0, 1], "m": pa.array([[("a", 1)], []], type=pa.map_(pa.string(), pa.int64()))}
    )
    duck.register("m", m)
    query = "SELECT k, m['a'] r FROM m ORDER BY k"
    assert_same_ordered(bt.sql(query, m=m).collect(), duck.sql(query))
