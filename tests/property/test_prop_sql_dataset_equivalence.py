"""Property: one query spelled as a `Dataset` chain and as SQL gives the same answer.

Batcher has two front ends over one plan. Each is checked against DuckDB on its own, in the
`tests/differential/` files, but nothing held them to *each other*: a filter literal the SQL
parser types differently, an aggregate it names or widens differently, or a NULL it
compares differently would pass both suites while a user porting ``ds.filter(...)`` to
``WHERE ...`` got a different table. This module generates a random table (NULLs, empty,
duplicates, NaN and ``-0.0``) and a random pipeline, renders the pipeline both ways, and
asserts the two results have the same schema, names and types, and the same multiset of
rows. An ``ORDER BY`` is compared in order, on its key.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pyarrow as pa
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import batcher as bt
from batcher import col

pytest.importorskip("batcher._native", reason="native engine not built")

pytestmark = [pytest.mark.property]

_SCHEMA = pa.schema([("k", pa.int64()), ("i", pa.int64()), ("f", pa.float64()), ("s", pa.string())])
_ints = st.one_of(st.none(), st.sampled_from([-3, -1, 0, 1, 2, 7]))
_floats = st.one_of(st.none(), st.sampled_from([0.0, -0.0, 1.5, -1.5, 2.5, float("nan")]))
_strs = st.one_of(st.none(), st.sampled_from(["a", "b", "", "ab"]))


@st.composite
def _table(draw: st.DrawFn) -> pa.Table:
    n = draw(st.integers(min_value=0, max_value=30))
    column = lambda values: draw(st.lists(values, min_size=n, max_size=n))  # noqa: E731
    return pa.table(
        {
            "k": pa.array(column(st.integers(0, 3)), pa.int64()),
            "i": pa.array(column(_ints), pa.int64()),
            "f": pa.array(column(_floats), pa.float64()),
            "s": pa.array(column(_strs), pa.string()),
        },
        schema=_SCHEMA,
    )


@dataclass(frozen=True)
class _Spelling:
    """One query, written for each front end."""

    dataset: object  # Callable[[bt.Dataset], bt.Dataset]
    sql: str


_OPS = {">": "__gt__", ">=": "__ge__", "<": "__lt__", "<=": "__le__", "=": "__eq__", "!=": "__ne__"}


@st.composite
def _predicate(draw: st.DrawFn) -> tuple[object, str]:
    """A WHERE clause as `(Expr, SQL text)`, or `(None, "")` for no filter."""
    kind = draw(st.sampled_from(["none", "int", "float", "str", "is_null", "not_null", "and"]))
    if kind == "none":
        return None, ""
    if kind == "int":
        op, v = draw(st.sampled_from(sorted(_OPS))), draw(st.sampled_from([-1, 0, 2]))
        return getattr(col("i"), _OPS[op])(v), f"i {op} {v}"
    if kind == "float":
        op, v = draw(st.sampled_from(sorted(_OPS))), draw(st.sampled_from([0.0, 1.5, -1.5]))
        return getattr(col("f"), _OPS[op])(v), f"f {op} {v!r}"
    if kind == "str":
        op, v = draw(st.sampled_from(["=", "!="])), draw(st.sampled_from(["a", ""]))
        return getattr(col("s"), _OPS[op])(v), f"s {op} '{v}'"
    if kind == "is_null":
        return col("i").is_null(), "i IS NULL"
    if kind == "not_null":
        return col("s").is_not_null(), "s IS NOT NULL"
    return (col("k") >= 1) & (col("i") < 7), "k >= 1 AND i < 7"


_AGGS = {
    "sum": ("SUM", lambda c: col(c).sum()),
    "count": ("COUNT", lambda c: col(c).count()),
    "min": ("MIN", lambda c: col(c).min()),
    "max": ("MAX", lambda c: col(c).max()),
    "mean": ("AVG", lambda c: col(c).mean()),
}


@st.composite
def _query(draw: st.DrawFn) -> _Spelling:
    pred, where_sql = draw(_predicate())

    def where(ds: bt.Dataset) -> bt.Dataset:
        return ds if pred is None else ds.filter(pred)

    where_clause = f" WHERE {where_sql}" if where_sql else ""
    shape = draw(st.sampled_from(["project", "group_agg", "global_agg", "distinct"]))
    if shape == "project":
        return _Spelling(
            lambda ds: where(ds).select("k", (col("i") * 2).alias("x"), "f", "s"),
            f"SELECT k, i * 2 AS x, f, s FROM t{where_clause}",
        )
    if shape == "distinct":
        return _Spelling(
            lambda ds: where(ds).select("k", "s").distinct(),
            f"SELECT DISTINCT k, s FROM t{where_clause}",
        )
    agg = draw(st.sampled_from(sorted(_AGGS)))
    column = draw(st.sampled_from(["i", "f"] if agg != "count" else ["i", "f", "s"]))
    sql_fn, build = _AGGS[agg]
    if shape == "group_agg":
        return _Spelling(
            lambda ds: where(ds).group_by("k").agg(v=build(column)),
            f"SELECT k, {sql_fn}({column}) AS v FROM t{where_clause} GROUP BY k",
        )
    return _Spelling(
        lambda ds: where(ds).agg(v=build(column)),
        f"SELECT {sql_fn}({column}) AS v FROM t{where_clause}",
    )


def _coerce(v: object) -> object:
    if isinstance(v, float):
        if math.isnan(v):
            return "<nan>"
        return round(v, 9) + 0.0  # `+ 0.0` folds -0.0 onto 0.0: equal under SQL `=`
    return v


def _rowset(table: pa.Table) -> list[tuple]:
    rows = [tuple(_coerce(v) for v in r.values()) for r in table.to_pylist()]
    return sorted(rows, key=lambda t: tuple((v is None, str(type(v)), str(v)) for v in t))


_PROP = settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


@_PROP
@given(_table(), _query())
def test_sql_and_dataset_spellings_agree(table: pa.Table, query: _Spelling) -> None:
    from_dataset = query.dataset(bt.from_arrow(table)).collect()
    from_sql = bt.sql(query.sql, t=bt.from_arrow(table)).collect()
    assert from_sql.schema.names == from_dataset.schema.names, query.sql
    assert from_sql.schema.types == from_dataset.schema.types, (
        f"{query.sql}\n sql={from_sql.schema}\n dataset={from_dataset.schema}"
    )
    assert _rowset(from_sql) == _rowset(from_dataset), (
        f"{query.sql}\n sql={_rowset(from_sql)}\n dataset={_rowset(from_dataset)}"
    )


@_PROP
@given(_table(), st.sampled_from(["k", "i", "f", "s"]), st.booleans())
def test_order_by_agrees_in_order(table: pa.Table, key: str, descending: bool) -> None:
    """An ``ORDER BY`` yields its key in the same sequence from both front ends.

    Compared in order on the key column alone: rows tied on the key may come out in either
    order, which neither spelling defines. ``NULLS LAST`` is spelled out on the SQL side to
    match `sort`'s default rather than relying on a dialect's.
    """
    direction = "DESC" if descending else "ASC"
    from_dataset = bt.from_arrow(table).sort(key, descending=descending, nulls_first=False)
    sql = f"SELECT * FROM t ORDER BY {key} {direction} NULLS LAST"
    from_sql = bt.sql(sql, t=bt.from_arrow(table))
    keys_sql = [_coerce(r[key]) for r in from_sql.collect().to_pylist()]
    keys_ds = [_coerce(r[key]) for r in from_dataset.collect().to_pylist()]
    assert keys_sql == keys_ds, f"{sql}\n sql={keys_sql}\n dataset={keys_ds}"
