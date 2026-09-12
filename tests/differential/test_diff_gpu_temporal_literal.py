"""A temporal column compared with a string literal, on the device translator.

`WHERE EventDate >= '2013-07-01'` is ClickBench q36 through q42 — seven of forty-three queries —
and until now the translator handed the comparison to the dataframe library as a `date32` column
against a Python `str`. pandas raises `TypeError: Invalid comparison`; cuDF raises its own. So
the query reached a device, started a worker, read its shard and *then* failed on a type
mismatch the driver could have seen.

The engine's rule is `bc-expr::eval::coerce`: cast the `Utf8` operand to the column's exact
temporal type with Arrow, which **nulls** what it cannot parse — hence `col == 'not-a-date'` is
unknown there rather than an error.

Reproducing that needed care in one specific place, and it is the reason this file exists rather
than a unit test: **the two Arrow implementations disagree**. Against a `Date32` column arrow-rs
accepts `'2013-07-15 12:30:05'` and takes the date part, and Arrow C++ raises `Failed to parse
string`. Both accept a bare `'2013-07-15'`, and for `Timestamp` they agree throughout. A
translation built on pyarrow's cast alone would therefore null a row the engine matches — the
right shape of column with the wrong rows in it.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher.api.terminal.gpu_backend.verify import compare_results
from batcher.core.gpu_plan import DfBackend, gpu_plan_ops
from batcher.core.gpu_plan.execute import run_chain

pytestmark = pytest.mark.differential

DATES = [dt.date(2013, 7, 1), dt.date(2013, 7, 15), dt.date(2013, 8, 1), None]
STAMPS = [
    dt.datetime(2013, 7, 1),
    dt.datetime(2013, 7, 15, 12, 30, 5),
    dt.datetime(2013, 8, 1),
    None,
]

#: Literals chosen to separate the two Arrow implementations, not to cover the format space.
#: The middle two are the ones a `Date32` cast disagrees about.
LITERALS = [
    "2013-07-15",
    "2013-07-15 12:30:05",
    "2013-07-15T12:30:05",
    "2013-07-01",
    "not-a-date",
    "2013/07/15",
]

#: The subset DuckDB and the engine agree on, which is the subset DuckDB can be an oracle for.
#: The other two are recorded as a divergence below rather than dropped — see
#: `test_the_engine_and_duckdb_differ_on_a_date_literal_arrow_cannot_parse`.
AGREED = ["2013-07-15", "2013-07-15 12:30:05", "2013-07-15T12:30:05", "2013-07-01"]

OPS = ["ge", "le", "gt", "lt", "eq", "ne"]
_SQL = {"ge": ">=", "le": "<=", "gt": ">", "lt": "<", "eq": "=", "ne": "<>"}


@pytest.fixture
def t(duck):
    tbl = pa.table({"d": pa.array(DATES, pa.date32()), "ts": pa.array(STAMPS, pa.timestamp("us"))})
    duck.register("t", tbl)
    return tbl


def _expr(column: str, op: str, literal: str):
    """`col(column) <op> literal`, built by the operator name the IR uses."""
    return getattr(bt.col(column), f"__{op}__")(literal)


@pytest.mark.parametrize("literal", AGREED)
@pytest.mark.parametrize("column", ["d", "ts"])
def test_the_engine_matches_duckdb(duck, t, column, literal):
    """The oracle first: this is the semantics being translated, not one this file invents."""
    out = bt.from_arrow(t).select(r=_expr(column, "ge", literal)).collect()
    assert_same(out, duck.sql(f"SELECT {column} >= '{literal}' AS r FROM t"))


@pytest.mark.parametrize("literal", ["not-a-date", "2013/07/15"])
def test_the_engine_and_duckdb_differ_on_a_date_literal_arrow_cannot_parse(duck, t, literal):
    """A **pre-existing engine/DuckDB divergence**, recorded rather than quietly excluded.

    It is not introduced by the device translation and is not changed by it — the translation's
    whole contract is to reproduce whatever the engine does, and it does, which is what every
    other test here checks. It surfaced because this file made DuckDB the oracle for a
    comparison nobody had put in front of it.

    Where they differ:

    * `'not-a-date'` — DuckDB raises `Conversion Error: invalid date field format`; the engine
      casts with Arrow, which nulls an unparseable value, so the comparison is SQL's unknown.
    * `'2013/07/15'` — DuckDB parses slash-separated dates; Arrow does not, so the engine nulls
      it and DuckDB compares it.

    Both are the *same* underlying difference: DuckDB has its own date-literal parser and the
    engine uses Arrow's. Neither is more correct, and nothing in the suites this repo runs
    depends on the gap; it is written down here so a future change to either side is a decision
    rather than a surprise.
    """
    got = bt.from_arrow(t).select(r=bt.col("d") >= literal).to_pydict()["r"]
    assert got == [None, None, None, None], "the engine nulls what Arrow cannot parse"
    try:
        answer = duck.sql(f"SELECT d >= '{literal}' AS r FROM t").arrow()
    except Exception:
        return  # DuckDB refuses it outright, which is the divergence for `'not-a-date'`
    rows = answer.read_all().column("r").to_pylist()
    assert rows != got, "the divergence this test records has closed; update it deliberately"


@pytest.mark.parametrize("literal", LITERALS)
@pytest.mark.parametrize("column", ["d", "ts"])
@pytest.mark.parametrize("op", OPS)
def test_the_device_translation_matches_the_engine(t, column, literal, op):
    """Same rows *and* the same column type, which is what `gpu_shadow_verify` checks at
    runtime and what this tier's defects have always been about."""
    ds = bt.from_arrow(t).select(r=_expr(column, op, literal))
    matched = gpu_plan_ops(ds._plan)
    assert matched is not None, "the comparison must reach the translator for this to test it"
    be = DfBackend(pd)
    translated = be.to_arrow(run_chain(t, matched[1], be))
    why = compare_results(translated, ds.collect())
    assert why is None, f"{column} {op} {literal!r}: {why}"


def test_a_datetime_literal_against_a_date_column_takes_its_date(t):
    """The case the two Arrow implementations disagree about, stated as the engine answers it."""
    got = bt.from_arrow(t).select(r=bt.col("d") == "2013-07-15 12:30:05").to_pydict()["r"]
    assert got == [False, True, False, None]


def test_an_unparseable_literal_is_unknown_and_not_an_error(t):
    """Arrow nulls what it cannot parse, so the comparison is SQL's unknown — every row.

    Not checked against DuckDB, which refuses the literal outright; see the divergence test.
    """
    out = bt.from_arrow(t).select(r=bt.col("d") >= "not-a-date").collect()
    assert out.column("r").to_pylist() == [None, None, None, None]


def test_an_unknown_predicate_filters_every_row(t):
    """`WHERE unknown` keeps nothing, which is where an all-null mask actually lands."""
    assert bt.from_arrow(t).filter(bt.col("d") >= "not-a-date").collect().num_rows == 0


@pytest.mark.parametrize("literal", ["2013-07-15", "not-a-date"])
def test_the_literal_may_be_on_either_side(t, literal):
    ds = bt.from_arrow(t).select(r=bt.lit(literal) <= bt.col("d"))
    matched = gpu_plan_ops(ds._plan)
    if matched is None:
        pytest.skip("the constant-folded form does not reach the translator")
    be = DfBackend(pd)
    assert compare_results(be.to_arrow(run_chain(t, matched[1], be)), ds.collect()) is None
