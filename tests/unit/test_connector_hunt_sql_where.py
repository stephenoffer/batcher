"""SQL predicate-pushdown generation must be correct and portable.

`batcher.io.predicate.to_sql_where` translates Kyber's pushed predicate IR into a
SQL ``WHERE`` fragment that every relational connector (Snowflake, BigQuery,
ClickHouse, Databricks, ADBC, ConnectorX, ODBC) sends *to the database*. A
mis-generated fragment does not read too many rows — it makes the database return
the *wrong* rows (or reject the query), which the engine's post-scan ``Filter``
cannot repair because the wrong rows already crossed the wire.

These pin two defects that produced exactly that:

- A temporal literal (``date`` / ``timestamp`` / ``time``) was emitted **unquoted**:
  ``d = 2021-01-15``, which SQL evaluates as the integer arithmetic
  ``2021 - 1 - 15`` (→ 2005), and ``t > 2020-09-13 12:26:40``, a syntax error.
- A non-finite float literal (``NaN`` / ``±Inf``) was emitted as a bare token
  ``col = nan`` / ``col < inf`` — invalid SQL in every targeted warehouse.

Pure control-plane string generation; no live database is required.
"""

from __future__ import annotations

import pytest

from batcher.io.predicate import to_sql_where


def _lit(kind: str, value: object) -> dict:
    return {"e": "lit", "value": {kind: value}}


def _col(name: str) -> dict:
    return {"e": "col", "name": name}


def _binary(op: str, left: dict, right: dict) -> dict:
    return {"e": "binary", "op": op, "left": left, "right": right}


@pytest.mark.unit
def test_date_literal_is_quoted_not_arithmetic():
    # 18642 days after the epoch is 2021-01-15. Emitted unquoted, SQL reads
    # `d = 2021-01-15` as `d = (2021 - 1 - 15)` = `d = 2005` — wrong rows.
    where = to_sql_where(_binary("eq", _col("d"), _lit("date", 18642)))
    assert where == "d = DATE '2021-01-15'"


@pytest.mark.unit
def test_timestamp_literal_is_quoted():
    # 1_600_000_000_000_000 micros = 2020-09-13 12:26:40. Unquoted this is a
    # SQL syntax error (`t > 2020-09-13 12:26:40`).
    where = to_sql_where(_binary("gt", _col("t"), _lit("timestamp", 1_600_000_000_000_000)))
    assert where == "t > TIMESTAMP '2020-09-13 12:26:40'"


@pytest.mark.unit
def test_time_literal_is_quoted():
    # noon = 12 * 3600 * 1e6 micros since midnight.
    where = to_sql_where(_binary("eq", _col("tm"), _lit("time", 12 * 3600 * 1_000_000)))
    assert where == "tm = TIME '12:00:00'"


@pytest.mark.unit
def test_flipped_date_literal_on_left_is_quoted():
    # literal-on-left: `DATE '2021-01-15' < d`  →  `d > DATE '2021-01-15'`.
    where = to_sql_where(_binary("lt", _lit("date", 18642), _col("d")))
    assert where == "d > DATE '2021-01-15'"


@pytest.mark.unit
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_float_is_not_pushed(bad: float):
    # No portable SQL spelling for NaN/±Inf: `col = nan` / `col < inf` is rejected
    # by Snowflake/BigQuery/ClickHouse. Leave it unpushed (engine Filter re-checks).
    assert to_sql_where(_binary("eq", _col("f"), _lit("float", bad))) is None


@pytest.mark.unit
def test_non_finite_float_leaves_only_its_own_conjunct_unpushed():
    # The non-finite term drops; the finite one still goes to the database. Widening a
    # conjunction is safe (the engine's Filter re-checks every row that comes back) and
    # the alternative is a full table extract because of one unspellable literal.
    pred = _binary(
        "and",
        _binary("eq", _col("a"), _lit("int", 1)),
        _binary("lt", _col("f"), _lit("float", float("inf"))),
    )
    assert to_sql_where(pred) == "a = 1"


@pytest.mark.unit
def test_non_finite_float_makes_a_disjunction_unpushable():
    # An OR is the opposite: keeping only `a = 1` would drop the rows the other side
    # matched, and no post-scan Filter can recover rows that never crossed the wire.
    pred = _binary(
        "or",
        _binary("eq", _col("a"), _lit("int", 1)),
        _binary("lt", _col("f"), _lit("float", float("inf"))),
    )
    assert to_sql_where(pred) is None


@pytest.mark.unit
def test_finite_literals_still_push():
    assert to_sql_where(_binary("eq", _col("i"), _lit("int", 5))) == "i = 5"
    assert to_sql_where(_binary("lt", _col("x"), _lit("float", 3.5))) == "x < 3.5"
    assert to_sql_where(_binary("eq", _col("b"), _lit("bool", True))) == "b = TRUE"
    # embedded quote stays escaped by doubling (injection-safe)
    assert to_sql_where(_binary("eq", _col("s"), _lit("str", "O'Brien"))) == "s = 'O''Brien'"
