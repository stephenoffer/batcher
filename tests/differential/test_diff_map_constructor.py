"""Differential tests for the `map(keys, values)` constructor against DuckDB.

The read side of maps (`.map.keys`/`.map.values`/`.map.entries`/`.map.get`) already worked on
a `Map` column arriving from Arrow. What was missing was any way to *build* one, so every
`map_*` call over a constructed map failed on its argument rather than on the function.

Three inputs raise rather than being coerced, and those are the cases worth pinning: each has
a plausible wrong answer a permissive implementation returns instead of an error — a null key
(which Arrow's non-nullable key field cannot hold), a duplicate key (first or last is a guess),
and key/value lists of different lengths (truncating silently drops data). A null *value* is
legal, which is the asymmetry a "reject all nulls" shortcut would get wrong.
"""

from __future__ import annotations

import duckdb
import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential


def _pairs(value):
    """A map value as a sorted list of pairs, however the engine spells it."""
    if value is None:
        return None
    if isinstance(value, dict):
        return sorted(value.items())
    return sorted((k, v) for k, v in value)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT map(['a','b'], [1,2]) AS m",
        "SELECT map([], []) AS m",
        "SELECT map(['a'], [NULL]) AS m",
        "SELECT map(['x','y','z'], [10,20,30]) AS m",
    ],
)
def test_map_constructor_matches_duckdb(duck, sql):
    """The literal forms, including the empty map and the null map, which differ."""
    want = _pairs(duck.sql(sql).fetchone()[0])
    got = _pairs(bt.sql(sql).collect().to_pydict()["m"][0])
    assert got == want


@pytest.mark.parametrize(
    ("sql", "fragment"),
    [
        ("SELECT map([NULL], [1]) AS m", "NULL"),
        ("SELECT map(['a','a'], [1,2]) AS m", "unique"),
        ("SELECT map(['a','b'], [1]) AS m", "align"),
    ],
)
def test_map_constructor_refuses_what_duckdb_refuses(duck, sql, fragment):
    """DuckDB raises on all three; so must Batcher, rather than guessing an answer."""
    with pytest.raises(duckdb.InvalidInputException):
        duck.sql(sql).fetchone()
    # Engine-side refusals surface through the FFI as `RuntimeError`, which is the
    # convention the rest of this directory asserts on.
    with pytest.raises(RuntimeError) as got:
        bt.sql(sql).collect()
    assert fragment.lower() in str(got.value).lower()


def test_a_constructed_map_reaches_the_read_side(duck):
    """The point of the constructor: `map_*` over a built map, which used to fail on the
    argument. Held against DuckDB rather than against itself."""
    t = pa.table({"k": pa.array([["a", "b"], ["c"]]), "v": pa.array([[1, 2], [3]])})
    duck.register("t", t)
    for sql in (
        "SELECT map_keys(map(k, v)) AS r FROM t",
        "SELECT map_values(map(k, v)) AS r FROM t",
        "SELECT cardinality(map(k, v)) AS r FROM t",
    ):
        assert_same(bt.sql(sql, t=t).collect(), duck.sql(sql))


def test_map_from_arrays_is_the_same_constructor():
    """Spark's spelling reaches the same node; DuckDB has no such name, so this is checked
    against `map(...)` rather than against the oracle."""
    t = pa.table({"k": pa.array([["a"]]), "v": pa.array([[1]])})
    a = bt.sql("SELECT map(k, v) AS m FROM t", t=t).collect().to_pydict()["m"]
    b = bt.sql("SELECT map_from_arrays(k, v) AS m FROM t", t=t).collect().to_pydict()["m"]
    assert a == b


def test_the_dataframe_constructor_matches_the_sql_one():
    t = pa.table({"k": pa.array([["a", "b"]]), "v": pa.array([[1, 2]])})
    sql = bt.sql("SELECT map(k, v) AS m FROM t", t=t).collect().to_pydict()["m"]
    df = (
        bt.from_arrow(t)
        .select(bt.map_from_arrays(bt.col("k"), bt.col("v")).alias("m"))
        .collect()
        .to_pydict()["m"]
    )
    assert df == sql


@pytest.mark.parametrize("keys", [["a", "b"], ["a"]])
def test_a_per_row_duplicate_is_not_a_column_wide_one(duck, keys):
    """The same key in two different rows is ordinary data. A shared uniqueness set would
    reject it, and no single-row test would notice."""
    t = pa.table({"k": pa.array([keys, keys]), "v": pa.array([list(range(len(keys)))] * 2)})
    duck.register("t", t)
    sql = "SELECT cardinality(map(k, v)) AS r FROM t"
    assert_same(bt.sql(sql, t=t).collect(), duck.sql(sql))


def test_a_null_list_row_yields_a_null_map(duck):
    """A null list on either side is a null map, as DuckDB's is — and distinct from the
    empty map, which is the pair an offsets bug collapses."""
    t = pa.table(
        {
            "k": pa.array([["a"], None, []], type=pa.list_(pa.string())),
            "v": pa.array([[1], None, []], type=pa.list_(pa.int64())),
        }
    )
    duck.register("t", t)
    sql = "SELECT cardinality(map(k, v)) AS r FROM t"
    assert_same(bt.sql(sql, t=t).collect(), duck.sql(sql))


def test_an_untyped_null_is_refused_and_that_is_a_deliberate_divergence():
    """`map(NULL, NULL)` — a *bare* SQL NULL, not a null list column.

    DuckDB answers a null map. Batcher refuses, because sqlglot types a bare `NULL` as
    `Int64` and there is no key or value type to build a `Map` field from; inventing one
    would put a guessed schema into the plan. The null *list row* case above is the one that
    matters in practice and it does match DuckDB, so this divergence is narrow — but it is a
    divergence, and it is pinned here rather than left for someone to rediscover.
    """
    with pytest.raises(RuntimeError) as got:
        bt.sql("SELECT map(NULL, NULL) AS m").collect()
    assert "list" in str(got.value).lower()
