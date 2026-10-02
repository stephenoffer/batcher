"""SQL translator restrictions lifted where the engine already had the capability.

Each case was refused (or, for two of them, answered wrongly) by the SQL front-end while the
engine could compute it:

* ``list_contains(l, v)`` with a per-row `v` — lowered to ``list_has_any(l, [v])``, with
  DuckDB's NULL for a NULL `v`.
* ``json_extract(j, path_col)`` / ``j ->> path_col`` — a per-row path runs through
  `StrFuncDyn`, which groups rows by path and calls the same kernel.
* ``list_sort(l, 'DESC')`` / ``array_sort(l, 'ASC', 'NULLS FIRST')`` — DuckDB's string
  direction was read as a boolean and **returned the ascending order**.
* ``ntile(1 + 1)`` / ``nth_value(x, 3 - 1)`` — a constant *expression* was refused as if it
  were per-row.
* ``sha2(s, 224 | 384 | 512)`` — Spark's other digest widths (DuckDB has no `sha2`, so
  Python's `hashlib` is the oracle).

Every comparison is positional on a unique `id`.
"""

from __future__ import annotations

import hashlib

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.differential


def _rows(table: pa.Table) -> list[tuple]:
    return [tuple(r.values()) for r in table.to_pylist()]


def _check(duck, query: str, **tables: pa.Table) -> None:
    for name, table in tables.items():
        duck.register(name, table)
    assert _rows(bt.sql(query, **tables).collect()) == duck.sql(query).fetchall()


_LISTS = pa.table(
    {
        "id": pa.array([1, 2, 3, 4, 5, 6, 7], pa.int64()),
        "l": pa.array([[1, 2], [1, None], [], None, [3], [4, 4], [1, 2]], pa.list_(pa.int64())),
        "x": pa.array([2, None, 1, 1, 4, 4, None], pa.int64()),
        "s": pa.array([["a"], ["b", None], [], None, ["c"], ["d"], ["e"]]),
        "y": pa.array(["a", "x", None, "a", "c", "d", "z"]),
    }
)


@pytest.mark.parametrize(
    "expr",
    ["list_contains(l, x)", "list_contains(l, x + 1)", "list_has(l, x)", "list_contains(s, y)"],
)
def test_list_contains_with_a_per_row_value_matches_duckdb(duck, expr):
    _check(duck, f"SELECT id, {expr} AS r FROM t ORDER BY id", t=_LISTS)


_DOCS = pa.table(
    {
        "id": pa.array([1, 2, 3, 4, 5, 6], pa.int64()),
        "j": [
            '{"a":"x","b":[1,2]}',
            '{"a":null}',
            '{"b":{"c":3}}',
            None,
            '{"a":1}',
            '{"b":[1,2]}',
        ],
        "p": ["$.a", "$.a", "$.b.c", "$.a", None, "$.b[1]"],
    }
)


@pytest.mark.parametrize(
    "expr", ["json_extract(j, p)", "json_extract_string(j, p)", "j ->> p", "j -> p"]
)
def test_json_extraction_with_a_per_row_path_matches_duckdb(duck, expr):
    _check(duck, f"SELECT id, {expr} AS r FROM t ORDER BY id", t=_DOCS)


_UNSORTED = pa.table(
    {"id": pa.array([1, 2, 3], pa.int64()), "l": pa.array([[3, None, 1, 2], [], None])}
)


@pytest.mark.parametrize(
    "expr",
    [
        "list_sort(l, 'DESC')",
        "list_sort(l, 'desc')",
        "array_sort(l, 'ASC', 'NULLS FIRST')",
        "array_sort(l, 'DESC', 'NULLS FIRST')",
        "array_sort(l, 'DESC', 'NULLS LAST')",
        "list_sort(l)",
    ],
)
def test_a_string_sort_direction_is_honoured(duck, expr):
    _check(duck, f"SELECT id, {expr} AS r FROM t ORDER BY id", t=_UNSORTED)


def test_a_constant_expression_is_a_constant_window_argument(duck):
    series = pa.table({"o": pa.array([1, 2, 3, 4, 5], pa.int64()), "v": [5, 4, 3, 2, 1]})
    query = (
        "SELECT o, ntile(1 + 1) OVER (ORDER BY o) AS a, nth_value(v, 3 - 1) OVER (ORDER BY o) "
        "AS b, lag(v, 3 - 2) OVER (ORDER BY o) AS c FROM t ORDER BY o"
    )
    _check(duck, query, t=series)


def test_a_per_row_window_argument_is_still_refused():
    series = pa.table({"o": [1, 2], "v": [1, 2]})
    with pytest.raises(NotImplementedError, match="constant integer"):
        bt.sql("SELECT ntile(v) OVER (ORDER BY o) FROM t", t=series)


@pytest.mark.parametrize("bits", [0, 224, 256, 384, 512])
def test_sha2_digest_widths_match_hashlib(bits):
    text = pa.table({"id": [1, 2, 3], "s": ["abc", "", None]})
    got = bt.sql(f"SELECT id, sha2(s, {bits}) AS h FROM t ORDER BY id", t=text).to_pydict()["h"]
    digest = getattr(hashlib, f"sha{bits or 256}")
    assert got == [digest(b"abc").hexdigest(), digest(b"").hexdigest(), None]


def test_an_unknown_sha2_width_is_refused():
    with pytest.raises(NotImplementedError, match="224, 256"):
        bt.sql("SELECT sha2(s, 128) FROM t", t=pa.table({"s": ["a"]}))


@pytest.mark.parametrize(
    "expr",
    [
        "array_slice(l, 2, 3)",
        "array_slice(l, -4, 1)",
        "array_slice(l, 1, -2)",
        "list_reverse_sort(l)",
        "list_reverse_sort(l, 'NULLS FIRST')",
        "array_reverse_sort(l, 'NULLS LAST')",
    ],
)
def test_list_calls_whose_sqlglot_node_lost_an_argument_match_duckdb(duck, expr):
    """sqlglot builds Spark's `slice(l, start, length)` node for DuckDB's `array_slice(l, b, e)`,
    so the inclusive end was read as a length (`[2, 3, 4]` for DuckDB's `[2, 3]`), and it
    drops `list_reverse_sort`'s null-order argument. Both are now parsed as plain calls.
    """
    _check(duck, f"SELECT id, {expr} AS r FROM t ORDER BY id", t=_UNSORTED)


def test_spark_slice_keeps_its_length_argument():
    """The Spark spelling keeps sqlglot's node and its meaning: a start and a *length*."""
    table = pa.table({"l": pa.array([[1, 2, 3, 4]])})
    got = bt.sql("SELECT slice(l, 2, 2) AS r FROM t", t=table, dialect="spark").to_pydict()
    assert got == {"r": [[2, 3]]}


_CASES = pa.table(
    {
        "id": pa.array([1, 2, 3], pa.int64()),
        "x": pa.array([5, 1, None], pa.int64()),
        "s": ["a", "b", "c"],
    }
)


@pytest.mark.parametrize(
    "expr",
    [
        "CASE WHEN x > 2 THEN NULL ELSE TRUE END",
        "CASE WHEN x > 2 THEN NULL ELSE x > 1 END",
        "CASE WHEN x > 2 THEN NULL WHEN x IS NULL THEN FALSE ELSE NULL END",
        "CASE WHEN x > 2 THEN NULL ELSE s END",
        "CASE WHEN x > 2 THEN NULL ELSE DATE '2024-01-01' END",
    ],
)
def test_an_untyped_null_branch_takes_the_type_of_its_siblings(duck, expr):
    """The untyped NULL is Int64 in the IR, so it turned a boolean CASE into 1/0 Int64 and
    failed outright beside a string or date branch. The declared schema, the executed one and
    DuckDB's must agree, and so must the values.
    """
    query = f"SELECT id, {expr} AS b FROM t ORDER BY id"
    duck.register("t", _CASES)
    want = duck.sql(query).arrow()
    want = want.read_all() if hasattr(want, "read_all") else want
    ds = bt.sql(query, t=_CASES)
    got = ds.collect()
    assert ds.schema.field("b").type == got.schema.field("b").type == want.schema.field("b").type
    assert got.column("b").to_pylist() == want.column("b").to_pylist()


def test_the_dataframe_null_literal_types_like_its_sibling_branch():
    ds = bt.from_pydict({"x": [5, 1]}).select(
        b=bt.when(bt.col("x") > 2).then(bt.lit(None)).otherwise(bt.lit(True))
    )
    assert ds.schema.field("b").type == pa.bool_()
    assert ds.to_pydict() == {"b": [None, True]}
