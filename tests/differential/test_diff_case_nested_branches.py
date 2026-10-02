"""A ``CASE`` whose branches are lists, structs or maps selects values; it never compares them.

A ``CASE`` with no ``ELSE`` (and a ``then(None)``) is lowered with a typed NULL,
``nullif(v, v)`` over one of its own branch values, and ``NULLIF`` went through the flat
equality kernel, which refuses nested types. So ``when(c).then(list_col)`` and SQL's
``CASE WHEN c THEN [1, 2] END`` failed with "Nested comparison: List(Int64) == List(Int64)"
while the same ``CASE`` with an explicit ``otherwise`` ran. DuckDB is the oracle for values;
the declared ``Dataset.schema`` must also equal the executed schema, since the typed NULL is
where the result type of a bare ``CASE`` comes from.

The second half covers the comparison that fix exposed: ``=``, ``<>``, ``<``, ``<=``, ``>``,
``>=``, ``IS [NOT] DISTINCT FROM`` and ``NULLIF`` over two lists, structs or maps. DuckDB's
nested semantics hold element-wise: a top-level null is null, a null *inside* a value equals
null and sorts last, ``-0.0`` equals ``0.0`` and NaN equals NaN and is greatest, and a list
that is a proper prefix of another sorts first.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

_LIST = pa.list_(pa.int64())
_STRUCT = pa.struct([("x", pa.int64()), ("y", pa.string())])
_MAP = pa.map_(pa.string(), pa.int64())


def _table(a, l, m, ll, s, mp) -> pa.Table:  # noqa: E741 -- `l` names the list column
    return pa.table(
        {
            "a": pa.array(a, pa.int64()),
            "l": pa.array(l, _LIST),
            "m": pa.array(m, _LIST),
            "ll": pa.array(ll, pa.large_list(pa.int64())),
            "s": pa.array(s, _STRUCT),
            "mp": pa.array(mp, _MAP),
        }
    )


_TABLES = {
    # nulls at every depth: a null list, a list holding a null, an empty list, a null
    # struct, a struct of nulls, a null map and an empty map
    "nested_nulls": _table(
        a=[1, None, 3, -2, 5],
        l=[[1, None], None, [], [4], [5, 6]],
        m=[[9], [8, 7], None, [None], []],
        ll=[[1], [2], None, [3], []],
        s=[{"x": 1, "y": "p"}, None, {"x": None, "y": None}, {"x": 4, "y": "q"}, None],
        mp=[[("k", 1)], None, [], [("j", 2)], [("k", None)]],
    ),
    "empty": _table(a=[], l=[], m=[], ll=[], s=[], mp=[]),
    "one_row": _table(
        a=[7], l=[[1, 2]], m=[[3]], ll=[[4]], s=[{"x": 1, "y": "z"}], mp=[[("a", 1)]]
    ),
    # every row selects the same branch, and the other arm is never taken
    "duplicates": _table(
        a=[2, 2, 2],
        l=[[1], [1], [1]],
        m=[None, None, None],
        ll=[[1], [1], [1]],
        s=[{"x": 1, "y": "a"}] * 3,
        mp=[[("k", 1)]] * 3,
    ),
}


@pytest.fixture(params=sorted(_TABLES))
def table(request) -> pa.Table:
    return _TABLES[request.param]


def _check(ours: bt.Dataset, expected) -> None:
    out = ours.collect()
    assert ours.schema == out.schema
    if any(_has_nested_float(f.type) for f in out.schema):
        _assert_same_nested(out, expected)
    else:
        assert_same(out, expected)


def _has_nested_float(t: pa.DataType) -> bool:
    return pa.types.is_nested(t) and any(
        pa.types.is_floating(t.field(i).type) or _has_nested_float(t.field(i).type)
        for i in range(t.num_fields)
    )


def _canon(v):
    """`v` with every NaN, at any depth, as one comparable token.

    `assert_same` canonicalizes a top-level NaN but not one inside a list, and `nan != nan`,
    so two identical `[nan]` cells would compare unequal there.
    """
    if isinstance(v, float) and v != v:
        return "<nan>"
    if isinstance(v, float) and v == 0.0:
        return 0.0
    if isinstance(v, (list, tuple)):
        return tuple(_canon(x) for x in v)
    if isinstance(v, dict):
        return tuple((k, _canon(x)) for k, x in v.items())
    return v


def _assert_same_nested(out: pa.Table, expected) -> None:
    """`assert_same` for results holding floats inside nested values: a row multiset."""
    duck = expected.to_arrow_table().select(out.column_names)
    ours = sorted((_canon(tuple(r.values())) for r in out.to_pylist()), key=repr)
    theirs = sorted((_canon(tuple(r.values())) for r in duck.to_pylist()), key=repr)
    assert ours == theirs, f"\nBatcher: {ours}\nDuckDB:  {theirs}"


@pytest.mark.parametrize("column", ["l", "ll", "s", "mp"])
def test_a_nested_case_without_else_matches_duckdb(duck, table, column):
    duck.register("t", table)
    ours = bt.from_arrow(table).select(r=bt.when(bt.col("a") > 1).then(bt.col(column)))
    _check(ours, duck.sql(f"SELECT CASE WHEN a > 1 THEN {column} END AS r FROM t"))


@pytest.mark.parametrize("column", ["l", "ll", "s", "mp"])
def test_a_null_then_is_typed_by_the_nested_else(duck, table, column):
    duck.register("t", table)
    ours = bt.from_arrow(table).select(
        r=bt.when(bt.col("a") > 1).then(None).otherwise(bt.col(column))
    )
    _check(ours, duck.sql(f"SELECT CASE WHEN a > 1 THEN NULL ELSE {column} END AS r FROM t"))


def test_a_multi_branch_list_case_matches_duckdb(duck, table):
    duck.register("t", table)
    ours = bt.from_arrow(table).select(
        r=bt.when(bt.col("a") > 2)
        .then(bt.col("l"))
        .when(bt.col("a") < 0)
        .then(bt.col("m"))
        .otherwise(bt.col("l")),
        bare=bt.when(bt.col("a") > 2).then(bt.col("l")).when(bt.col("a") < 0).then(bt.col("m")),
    )
    _check(
        ours,
        duck.sql(
            "SELECT CASE WHEN a > 2 THEN l WHEN a < 0 THEN m ELSE l END AS r, "
            "CASE WHEN a > 2 THEN l WHEN a < 0 THEN m END AS bare FROM t"
        ),
    )


@pytest.mark.parametrize(
    "query",
    [
        "SELECT CASE WHEN a > 1 THEN [1, 2] END AS r FROM t",
        "SELECT CASE WHEN a > 1 THEN [1, 2] ELSE [3] END AS r FROM t",
        "SELECT CASE WHEN a > 1 THEN l END AS r FROM t",
        "SELECT CASE WHEN a > 1 THEN NULL ELSE l END AS r FROM t",
        "SELECT CASE WHEN a > 1 THEN s END AS r FROM t",
        "SELECT CASE WHEN a > 1 THEN mp END AS r FROM t",
        "SELECT CASE WHEN a IS NULL THEN ll END AS r FROM t",
        # a mask no row satisfies, so every value is the typed NULL
        "SELECT CASE WHEN a > 100 THEN l END AS r FROM t",
    ],
)
def test_a_nested_sql_case_matches_duckdb(duck, table, query):
    duck.register("t", table)
    _check(bt.sql(query, t=bt.from_arrow(table)), duck.sql(query))


def test_an_all_false_mask_is_a_typed_null_column(table):
    ours = bt.from_arrow(table).select(r=bt.when(bt.col("a") > 100).then(bt.col("l")))
    out = ours.collect()
    assert out.schema.field("r").type == _LIST
    assert out.column("r").null_count == table.num_rows


# --- comparisons of nested values ------------------------------------------------------

_FLIST = pa.list_(pa.float64())
_NAN = float("nan")


def _pairs(l, m, f, g, s, t, mp, mq) -> pa.Table:  # noqa: E741 -- `l` names the list column
    return pa.table(
        {
            "l": pa.array(l, _LIST),
            "m": pa.array(m, _LIST),
            "f": pa.array(f, _FLIST),
            "g": pa.array(g, _FLIST),
            "s": pa.array(s, _STRUCT),
            "t": pa.array(t, _STRUCT),
            "mp": pa.array(mp, _MAP),
            "mq": pa.array(mq, _MAP),
        }
    )


_PAIRS = {
    # row by row: equal with a nested null; differ in the last element; a proper prefix; a
    # top-level null; two empties; a nested null against a value; a null element against empty
    "pairs": _pairs(
        l=[[1, None], [1, 2], [1, 2], None, [], [1, None], [None]],
        m=[[1, None], [1, 3], [1, 2, 0], [1], [], [1, 2], []],
        f=[[-0.0], [_NAN], [_NAN], [1.5], [None], [], [2.0]],
        g=[[0.0], [_NAN], [float("inf")], None, [None], [1.0], [1.0]],
        s=[
            {"x": None, "y": None},
            {"x": 1, "y": "a"},
            {"x": 1, "y": None},
            None,
            {"x": 1, "y": "b"},
            {"x": None, "y": "a"},
            {"x": 3, "y": "c"},
        ],
        t=[
            {"x": None, "y": None},
            {"x": 2, "y": "a"},
            {"x": 1, "y": "a"},
            {"x": 1, "y": "a"},
            {"x": 1, "y": "a"},
            {"x": 1, "y": "a"},
            {"x": 3, "y": "c"},
        ],
        mp=[[("k", 1)], [("k", 1), ("j", 2)], [("k", None)], None, [], [("a", 1)], [("b", 1)]],
        mq=[[("k", 1)], [("j", 2), ("k", 1)], [("k", None)], [], [], [("a", 2)], [("a", 1)]],
    ),
    "empty": _pairs(l=[], m=[], f=[], g=[], s=[], t=[], mp=[], mq=[]),
}

_SIDES = [("l", "m"), ("f", "g"), ("s", "t"), ("mp", "mq")]


@pytest.fixture(params=sorted(_PAIRS))
def pairs(request) -> pa.Table:
    return _PAIRS[request.param]


@pytest.mark.parametrize(("left", "right"), _SIDES)
def test_nested_comparisons_match_duckdb(duck, pairs, left, right):
    duck.register("t", pairs)
    a, b = bt.col(left), bt.col(right)
    ours = bt.from_arrow(pairs).select(
        eq=a == b, ne=a != b, lt=a < b, le=a <= b, gt=a > b, ge=a >= b, same=a == a
    )
    _check(
        ours,
        duck.sql(
            f"SELECT {left} = {right} AS eq, {left} <> {right} AS ne, {left} < {right} AS lt, "
            f"{left} <= {right} AS le, {left} > {right} AS gt, {left} >= {right} AS ge, "
            f"{left} = {left} AS same FROM t"
        ),
    )


@pytest.mark.parametrize(("left", "right"), _SIDES)
def test_nested_sql_comparisons_match_duckdb(duck, pairs, left, right):
    duck.register("t", pairs)
    query = (
        f"SELECT {left} = {right} AS eq, {left} <> {right} AS ne, {left} < {right} AS lt, "
        f"{left} >= {right} AS ge, {left} IS DISTINCT FROM {right} AS dist, "
        f"{left} IS NOT DISTINCT FROM {right} AS same, NULLIF({left}, {right}) AS n FROM t"
    )
    _check(bt.sql(query, t=bt.from_arrow(pairs)), duck.sql(query))


@pytest.mark.parametrize(("left", "right"), _SIDES)
def test_null_safe_nested_equality_matches_duckdb(duck, pairs, left, right):
    duck.register("t", pairs)
    ours = bt.from_arrow(pairs).select(same=bt.col(left).eq_missing(bt.col(right)))
    _check(ours, duck.sql(f"SELECT {left} IS NOT DISTINCT FROM {right} AS same FROM t"))


@pytest.mark.parametrize(("left", "right"), _SIDES)
def test_a_nested_equality_filters_like_duckdb(duck, pairs, left, right):
    duck.register("t", pairs)
    query = f"SELECT {left}, {right} FROM t WHERE {left} = {right} OR {left} > {right}"
    _check(bt.sql(query, t=bt.from_arrow(pairs)), duck.sql(query))
    ours = bt.from_arrow(pairs).filter(bt.col(left) == bt.col(right)).select(left, right)
    _check(ours, duck.sql(f"SELECT {left}, {right} FROM t WHERE {left} = {right}"))


def test_nested_literal_comparison_matches_duckdb(duck, pairs):
    duck.register("t", pairs)
    query = (
        "SELECT l = [1, 2] AS eq, l < [1, 2] AS lt, l <> [] AS ne, l = NULL AS n, "
        "s IS DISTINCT FROM NULL AS d FROM t"
    )
    _check(bt.sql(query, t=bt.from_arrow(pairs)), duck.sql(query))


def test_comparing_different_nested_types_raises():
    """``[1] = [1.0]`` is true in DuckDB, which unifies the element types; that is not done."""
    ours = bt.from_arrow(_PAIRS["pairs"]).select(r=bt.col("l") == bt.col("f"))
    with pytest.raises(bt.BatcherError, match="cannot compare"):
        ours.collect()
