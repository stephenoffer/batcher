"""Differential tests (vs DuckDB) for the `temporal_extra` sargability rewrites.

Every query here is one the optimizer *rewrites*: a `date_trunc`/`strftime` comparison
becomes a bare-column range, a Date→Timestamp cast disappears, a temporal literal folds.
The rewrite is only allowed to be faster, never different — so each is run against
DuckDB on the same data, including the edges the rewrites turn on: NULLs, an empty
relation, leap days, year/month boundaries, and the *non-monotone* extractions
(`month`/`day`/`quarter`) that must **not** be turned into a range.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, lit
from batcher.kyber.rules.extra import temporal_extra as _temporal_extra  # noqa: F401  (registers)

_DATES = [
    dt.date(2020, 12, 31),  # year boundary (below)
    dt.date(2021, 1, 1),  # year boundary (on)
    dt.date(2021, 2, 28),  # month boundary (non-leap February)
    dt.date(2021, 3, 1),  # month boundary (on)
    dt.date(2021, 3, 5),
    dt.date(2021, 3, 31),  # month boundary (last day)
    dt.date(2024, 2, 29),  # leap day
    None,  # NULL
]
_TIMES = [
    dt.datetime(2020, 12, 31, 23, 59, 59),
    dt.datetime(2021, 1, 1, 0, 0, 0),
    dt.datetime(2021, 2, 28, 12, 0, 0),
    dt.datetime(2021, 3, 1, 0, 0, 0),
    dt.datetime(2021, 3, 5, 4, 5, 6),
    dt.datetime(2021, 3, 31, 23, 59, 59),
    dt.datetime(2024, 2, 29, 6, 0, 0),
    None,
]


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "d": pa.array(_DATES, pa.date32()),
            "ts": pa.array(_TIMES, pa.timestamp("us")),
            "v": list(range(len(_DATES))),
        }
    )
    duck.register("t", tbl)
    return tbl


@pytest.fixture
def empty(duck):
    tbl = pa.table(
        {
            "d": pa.array([], pa.date32()),
            "ts": pa.array([], pa.timestamp("us")),
            "v": pa.array([], pa.int64()),
        }
    )
    duck.register("e", tbl)
    return tbl


# --- date_trunc inequalities -----------------------------------------------------


@pytest.mark.parametrize(
    ("unit", "op", "sql_op"),
    [
        ("month", "ge", ">="),
        ("month", "le", "<="),
        ("month", "gt", ">"),
        ("month", "lt", "<"),
        ("year", "ge", ">="),
        ("year", "lt", "<"),
        ("day", "le", "<="),
        ("day", "gt", ">"),
    ],
)
def test_date_trunc_inequality_on_timestamp(duck, t, unit, op, sql_op):
    from conftest import assert_same

    bound = {"month": dt.datetime(2021, 3, 1), "year": dt.datetime(2021, 1, 1)}.get(
        unit, dt.datetime(2021, 3, 5)
    )
    trunc = col("ts").dt.truncate(unit)
    expr = {
        "ge": trunc >= lit(bound),
        "le": trunc <= lit(bound),
        "gt": trunc > lit(bound),
        "lt": trunc < lit(bound),
    }[op]
    out = bt.from_arrow(t).filter(expr).select("v").collect()
    assert_same(
        out,
        duck.sql(f"SELECT v FROM t WHERE date_trunc('{unit}', ts) {sql_op} TIMESTAMP '{bound}'"),
    )


def test_date_trunc_inequality_on_date_column(duck, t):
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .filter(col("d").dt.truncate("month") >= lit(dt.date(2021, 3, 1)))
        .select("v")
        .collect()
    )
    assert_same(out, duck.sql("SELECT v FROM t WHERE date_trunc('month', d) >= DATE '2021-03-01'"))


def test_date_trunc_inequality_over_empty_input(duck, empty):
    from conftest import assert_same

    out = (
        bt.from_arrow(empty)
        .filter(col("ts").dt.truncate("year") > lit(dt.datetime(2021, 1, 1)))
        .select("v")
        .collect()
    )
    assert_same(
        out, duck.sql("SELECT v FROM e WHERE date_trunc('year', ts) > TIMESTAMP '2021-01-01'")
    )


def test_date_trunc_unaligned_literal_matches_nothing(duck, t):
    from conftest import assert_same

    # Not a month boundary: no row's truncation can equal it. The rule leaves the
    # comparison alone; the engine must still agree with DuckDB.
    out = (
        bt.from_arrow(t)
        .filter(col("ts").dt.truncate("month") >= lit(dt.datetime(2021, 3, 15)))
        .select("v")
        .collect()
    )
    assert_same(
        out, duck.sql("SELECT v FROM t WHERE date_trunc('month', ts) >= TIMESTAMP '2021-03-15'")
    )


# --- strftime comparisons --------------------------------------------------------


@pytest.mark.parametrize(
    ("fmt", "value"),
    [("%Y", "2021"), ("%Y-%m", "2021-03"), ("%Y-%m-%d", "2024-02-29")],
)
def test_strftime_equality(duck, t, fmt, value):
    from conftest import assert_same

    out = bt.from_arrow(t).filter(col("d").dt.strftime(fmt) == lit(value)).select("v").collect()
    assert_same(out, duck.sql(f"SELECT v FROM t WHERE strftime(d, '{fmt}') = '{value}'"))


@pytest.mark.parametrize(("op", "sql_op"), [("ge", ">="), ("le", "<="), ("gt", ">"), ("lt", "<")])
def test_strftime_inequality_on_timestamp(duck, t, op, sql_op):
    from conftest import assert_same

    formatted = col("ts").dt.strftime("%Y-%m")
    expr = {
        "ge": formatted >= lit("2021-03"),
        "le": formatted <= lit("2021-03"),
        "gt": formatted > lit("2021-03"),
        "lt": formatted < lit("2021-03"),
    }[op]
    out = bt.from_arrow(t).filter(expr).select("v").collect()
    assert_same(out, duck.sql(f"SELECT v FROM t WHERE strftime(ts, '%Y-%m') {sql_op} '2021-03'"))


def test_strftime_year_boundary_and_leap_day(duck, t):
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .filter(
            (col("d").dt.strftime("%Y") == lit("2021"))
            | (col("d").dt.strftime("%Y-%m-%d") == lit("2024-02-29"))
        )
        .select("v")
        .collect()
    )
    assert_same(
        out,
        duck.sql(
            "SELECT v FROM t WHERE strftime(d, '%Y') = '2021' "
            "OR strftime(d, '%Y-%m-%d') = '2024-02-29'"
        ),
    )


def test_strftime_impossible_literal(duck, t):
    from conftest import assert_same

    # `2021-02-30` never renders — the rule refuses it, and both engines return nothing.
    out = (
        bt.from_arrow(t)
        .filter(col("d").dt.strftime("%Y-%m-%d") == lit("2021-02-30"))
        .select("v")
        .collect()
    )
    assert_same(out, duck.sql("SELECT v FROM t WHERE strftime(d, '%Y-%m-%d') = '2021-02-30'"))


# --- the non-monotone extractions (must not have become a range) -----------------


@pytest.mark.parametrize(
    ("expr_fn", "sql"),
    [
        (lambda: col("d").dt.month() == lit(3), "month(d) = 3"),
        (lambda: col("d").dt.day() == lit(1), "day(d) = 1"),
        (lambda: col("d").dt.quarter() == lit(1), "quarter(d) = 1"),
        (lambda: col("d").dt.year() == lit(2021), "year(d) = 2021"),
    ],
)
def test_extraction_predicates(duck, t, expr_fn, sql):
    from conftest import assert_same

    out = bt.from_arrow(t).filter(expr_fn()).select("v").collect()
    assert_same(out, duck.sql(f"SELECT v FROM t WHERE {sql}"))


# --- Date → Timestamp cast -------------------------------------------------------


@pytest.mark.parametrize(("op", "sql_op"), [("ge", ">="), ("lt", "<"), ("eq", "="), ("ne", "!=")])
def test_date_cast_comparison(duck, t, op, sql_op):
    from conftest import assert_same

    cast = col("d").cast("timestamp")
    bound = lit(dt.datetime(2021, 3, 1))
    expr = {"ge": cast >= bound, "lt": cast < bound, "eq": cast == bound, "ne": cast != bound}[op]
    out = bt.from_arrow(t).filter(expr).select("v").collect()
    assert_same(
        out,
        duck.sql(f"SELECT v FROM t WHERE CAST(d AS TIMESTAMP) {sql_op} TIMESTAMP '2021-03-01'"),
    )


# --- nested truncation + constant folding ----------------------------------------


def test_nested_date_trunc_projection(duck, t):
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .select(col("ts").dt.truncate("day").dt.truncate("year").alias("r"), col("v"))
        .collect()
    )
    assert_same(out, duck.sql("SELECT date_trunc('year', date_trunc('day', ts)) AS r, v FROM t"))


def test_repeated_date_trunc_projection(duck, t):
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .select(col("ts").dt.truncate("month").dt.truncate("month").alias("r"), col("v"))
        .collect()
    )
    assert_same(out, duck.sql("SELECT date_trunc('month', date_trunc('month', ts)) AS r, v FROM t"))


def test_fold_date_func_of_literal(duck, t):
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .select(
            lit(dt.date(2024, 2, 29)).dt.year().alias("y"),
            lit(dt.date(2024, 2, 29)).dt.quarter().alias("q"),
            lit(dt.date(2024, 2, 29)).dt.dayofyear().alias("doy"),
            col("v"),
        )
        .collect()
    )
    assert_same(
        out,
        duck.sql(
            "SELECT year(DATE '2024-02-29') AS y, quarter(DATE '2024-02-29') AS q, "
            "dayofyear(DATE '2024-02-29') AS doy, v FROM t"
        ),
    )


def test_fold_date_offset_of_literal(duck, t):
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .select(lit(dt.date(2024, 2, 28)).dt.offset_by("1d").alias("r"), col("v"))
        .collect()
    )
    assert_same(
        out, duck.sql("SELECT CAST(DATE '2024-02-28' + INTERVAL 1 DAY AS DATE) AS r, v FROM t")
    )


def test_fold_temporal_literal_comparison(duck, t):
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .filter(lit(dt.date(2021, 1, 1)) < lit(dt.date(2021, 3, 5)))
        .select("v")
        .collect()
    )
    assert_same(out, duck.sql("SELECT v FROM t WHERE DATE '2021-01-01' < DATE '2021-03-05'"))

    out = (
        bt.from_arrow(t)
        .filter(lit(dt.date(2021, 3, 5)) < lit(dt.date(2021, 1, 1)))
        .select("v")
        .collect()
    )
    assert_same(out, duck.sql("SELECT v FROM t WHERE DATE '2021-03-05' < DATE '2021-01-01'"))
