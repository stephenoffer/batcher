"""A wave of SQL-surface defects found by differential fuzzing, one section each.

Every case here failed or answered differently from DuckDB before its fix. They are
grouped by the layer they were fixed at rather than split across files, because each is a
small, independent rule and the file reads as the list of what the translator now gets
right.
"""

from __future__ import annotations

import datetime as dt
import decimal

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential


def _table() -> pa.Table:
    return pa.table(
        {
            "i": pa.array([0, 1, 2, None], pa.int64()),
            "j": pa.array([10, 20, 30, 40], pa.int64()),
            "f": pa.array([0.0, 1.5, 0.0, None], pa.float64()),
            "b": pa.array([True, False, None, True], pa.bool_()),
            "g": pa.array(["x", "y", "x", None], pa.string()),
            "d": pa.array(
                [dt.date(2020, 1, 1), dt.date(2020, 1, 3), None, dt.date(2020, 1, 5)],
                pa.date32(),
            ),
            "ts": pa.array(
                [
                    dt.datetime(2020, 1, 1),
                    dt.datetime(2020, 1, 3),
                    None,
                    dt.datetime(2020, 1, 5),
                ],
                pa.timestamp("us"),
            ),
        }
    )


# --- a number used as a condition -------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT i FROM t WHERE i",
        "SELECT i FROM t WHERE (i)",
        "SELECT i FROM t WHERE NOT i",
        "SELECT i FROM t WHERE i AND f",
        "SELECT i FROM t WHERE i OR b",
        "SELECT i FROM t WHERE f",
    ],
)
def test_a_numeric_column_reads_as_a_condition(duck, sql):
    """`WHERE flag` over an integer is ordinary SQL; the filter takes a boolean."""
    table = _table()
    duck.register("t", table)
    assert_same(bt.sql(sql, t=table).collect(), duck.sql(sql))


# --- row (tuple) membership -------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT count(*) AS c FROM t WHERE (i, j) IN ((1, 20), (2, 30))",
        "SELECT count(*) AS c FROM t WHERE (i, j) NOT IN ((1, 20))",
        "SELECT i, (i, j) IN ((1, 20)) AS r FROM t",
        "SELECT i, (i, j) IN ((NULL, NULL)) AS r FROM t",
    ],
)
def test_row_membership_uses_null_safe_equality(duck, sql):
    """DuckDB compares a row's elements null-safely, so the answer is never NULL."""
    table = _table()
    duck.register("t", table)
    assert_same(bt.sql(sql, t=table).collect(), duck.sql(sql))


# --- set operations aligned by name -----------------------------------------------


def test_union_by_name_pairs_columns_by_name(duck):
    """`BY NAME` pairs on names, not positions; a name only one side has fills with NULL."""
    left = pa.table({"i": pa.array([1, 2]), "g": pa.array(["a", "b"])})
    right = pa.table({"g": pa.array(["z"]), "k": pa.array([9])})
    duck.register("t", left)
    duck.register("u", right)
    for sql in (
        "SELECT i, g FROM t UNION ALL BY NAME SELECT g, i FROM t",
        "SELECT i, g FROM t UNION BY NAME SELECT * FROM u",
    ):
        assert_same(bt.sql(sql, t=left, u=right).collect(), duck.sql(sql))


# --- a select-list alias named in WHERE -------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT i AS z FROM t WHERE z > 0",
        "SELECT i * 2 AS z FROM t WHERE z > 2",
        "SELECT i AS j FROM t WHERE j > 0",
    ],
)
def test_a_select_alias_resolves_in_where(duck, sql):
    """DuckDB resolves one, and a real column of the same name still wins."""
    table = _table()
    duck.register("t", table)
    assert_same(bt.sql(sql, t=table).collect(), duck.sql(sql))


# --- aggregates over types the kernels do not take directly -----------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT sum(CAST(i AS INTEGER)) AS r FROM t",
        "SELECT avg(CAST(i AS SMALLINT)) AS r FROM t",
        "SELECT min(CAST(i AS INTEGER)) AS r FROM t",
        "SELECT max(CAST(i AS TINYINT)) AS r FROM t",
        "SELECT sum(CAST(f AS REAL)) AS r FROM t",
        "SELECT stddev(CAST(i AS INTEGER)) AS r FROM t",
        "SELECT sum(CAST(NULL AS INTEGER)) AS r FROM t",
        "SELECT g, sum(CAST(i AS INTEGER)) AS r FROM t GROUP BY g",
        "SELECT sum(b) AS r FROM t",
        "SELECT g, sum(b) AS r FROM t GROUP BY g",
        "SELECT avg(d) AS r FROM t",
        "SELECT avg(ts) AS r FROM t",
        "SELECT g, avg(ts) AS r FROM t GROUP BY g",
    ],
)
def test_an_aggregate_widens_the_input_types_sql_admits(duck, sql):
    """A narrow width introduced *inside* the query never reached the FFI normalization."""
    table = _table()
    duck.register("t", table)
    assert_same(bt.sql(sql, t=table).collect(), duck.sql(sql))


# --- window aggregates over the types SQL admits ----------------------------------


def _window_table() -> pa.Table:
    return pa.table(
        {
            "k": pa.array([1, 1, 2, 2], pa.int64()),
            "o": pa.array([1, 2, 1, 2], pa.int64()),
            "dec": pa.array(
                [decimal.Decimal("1.50"), decimal.Decimal("2.50"), None, decimal.Decimal("4.50")],
                pa.decimal128(7, 2),
            ),
            "d": pa.array(
                [dt.date(2020, 1, 1), dt.date(2020, 1, 2), None, dt.date(2020, 1, 4)],
                pa.date32(),
            ),
            "ts": pa.array(
                [
                    dt.datetime(2020, 1, 1),
                    dt.datetime(2020, 1, 2),
                    None,
                    dt.datetime(2020, 1, 4),
                ],
                pa.timestamp("us"),
            ),
            "flag": pa.array([True, False, None, True], pa.bool_()),
        }
    )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT k, sum(dec) OVER (PARTITION BY k ORDER BY o ROWS UNBOUNDED PRECEDING) AS w FROM t",
        "SELECT k, avg(dec) OVER (PARTITION BY k ORDER BY o "
        "ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS w FROM t",
        "SELECT k, min(d) OVER (PARTITION BY k ORDER BY o ROWS UNBOUNDED PRECEDING) AS w FROM t",
        "SELECT k, max(d) OVER (PARTITION BY k ORDER BY o "
        "ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS w FROM t",
        "SELECT k, min(ts) OVER (PARTITION BY k ORDER BY o ROWS UNBOUNDED PRECEDING) AS w FROM t",
        "SELECT k, max(dec) OVER (PARTITION BY k ORDER BY o ROWS UNBOUNDED PRECEDING) AS w FROM t",
        "SELECT k, sum(flag) OVER (PARTITION BY k ORDER BY o) AS w FROM t",
        "SELECT k, avg(ts) OVER (PARTITION BY k) AS w FROM t",
        "SELECT k, avg(d) OVER (PARTITION BY k ORDER BY o) AS w FROM t",
        "SELECT k, avg(ts) OVER (PARTITION BY k ORDER BY o ROWS UNBOUNDED PRECEDING) AS w FROM t",
    ],
)
def test_a_framed_window_takes_the_types_the_frameless_one_does(duck, sql):
    """`min(order_date) OVER (… ROWS …)` answered nothing while the frameless form did."""
    table = _window_table()
    duck.register("t", table)
    assert_same(bt.sql(sql, t=table).collect(), duck.sql(sql))


def test_a_framed_min_over_a_decimal_keeps_its_type():
    """Widening MIN/MAX into the float kernel would answer DOUBLE for a DECIMAL input."""
    table = _window_table()
    sql = "SELECT k, min(dec) OVER (PARTITION BY k ORDER BY o ROWS UNBOUNDED PRECEDING) AS w FROM t"
    schema = bt.sql(sql, t=table).schema
    assert pa.types.is_decimal(schema.field("w").type)


# --- list index origin -------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id, list_extract(a, 0) AS r FROM l",
        "SELECT id, list_extract(a, 1) AS r FROM l",
        "SELECT id, list_extract(a, -1) AS r FROM l",
        "SELECT id, list_extract(a, -2) AS r FROM l",
        "SELECT id, list_extract(a, 9) AS r FROM l",
        "SELECT id, a[0] AS r FROM l",
        "SELECT id, a[1] AS r FROM l",
        "SELECT id, a[-1] AS r FROM l",
    ],
)
def test_a_list_index_is_one_based_and_counts_back_from_the_end(duck, sql):
    """Index 0 names no element (NULL); a negative one counts from the end."""
    table = pa.table(
        {
            "id": pa.array([1, 2, 3], pa.int64()),
            "a": pa.array([[1, 2, 3], [9], None], pa.list_(pa.int64())),
        }
    )
    duck.register("l", table)
    assert_same(bt.sql(sql, l=table).collect(), duck.sql(sql))


# --- `len` is defined on more than strings ----------------------------------------


def test_len_reads_a_list_or_a_map_by_its_type(duck):
    """Dispatching on the name alone sent a list column into the string kernel."""
    table = pa.table(
        {
            "id": pa.array([1, 2, 3], pa.int64()),
            "a": pa.array([[1, 2, 3], [], None], pa.list_(pa.int64())),
            "s": pa.array(["ab", "c", None], pa.string()),
        }
    )
    duck.register("l", table)
    for sql in (
        "SELECT id, len(a) AS r FROM l",
        "SELECT id, length(a) AS r FROM l",
        "SELECT id, len(s) AS r FROM l",
    ):
        assert_same(bt.sql(sql, l=table).collect(), duck.sql(sql))


# --- a list index computed per row -------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id, a[i] AS r FROM l",
        "SELECT id, a[z] AS r FROM l",
        "SELECT id, a[i + 1] AS r FROM l",
        "SELECT id, list_extract(a, i) AS r FROM l",
        "SELECT id, list_extract(a, z) AS r FROM l",
        "SELECT id, a[1] AS r FROM l",
        "SELECT id, a[-1] AS r FROM l",
        "SELECT id, a[0] AS r FROM l",
    ],
)
def test_a_list_index_may_be_a_column(duck, sql):
    """`a[i]` over an index column was refused; the constant form must not change.

    The index column carries 0 (out of range), a negative (from the end), a NULL and a
    value past the end, so the whole addressing rule is exercised per row rather than
    folded at plan time.
    """
    table = pa.table(
        {
            "id": pa.array([1, 2, 3, 4], pa.int64()),
            "a": pa.array([[1, 2, 3], [9], None, [4, 5]], pa.list_(pa.int64())),
            "i": pa.array([1, 2, 1, -1], pa.int64()),
            "z": pa.array([0, -1, 3, None], pa.int64()),
        }
    )
    duck.register("l", table)
    assert_same(bt.sql(sql, l=table).collect(), duck.sql(sql))


def test_a_per_row_list_index_is_independent_of_partitioning():
    """Elementwise, so splitting the input must change nothing."""
    table = pa.table(
        {
            "a": pa.array([[1, 2, 3], [9], None, [4, 5]] * 4, pa.list_(pa.int64())),
            "i": pa.array([1, 2, 1, -1] * 4, pa.int64()),
        }
    )
    ds = bt.sql("SELECT a[i] AS r FROM l", l=table)
    assert ds.collect().to_pydict() == ds.repartition(4).collect().to_pydict()


# --- a struct field written with a dot ---------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id, st.a AS r FROM l",
        "SELECT id, st.b AS r FROM l",
        "SELECT id, (st).a AS r FROM l",
        "SELECT id, l.st.a AS r FROM l",
        "SELECT id, n.p.q AS r FROM l",
        "SELECT id, l.n.p.q AS r FROM l",
        "SELECT id, struct_extract(st, 'a') AS r FROM l",
        "SELECT id, l.id AS r FROM l",
    ],
)
def test_a_dotted_struct_field_resolves(duck, sql):
    """`st.a` and a qualified column `t.a` parse identically; the schema decides which.

    Only `struct_extract(st, 'a')` and `st['a']` worked — the spelling every SQL dialect
    documents first did not, because the qualified reading looked for a *column* `a`.
    A table qualifier in front of the struct (`l.st.a`) and a nested chain (`n.p.q`) must
    keep working too.
    """
    table = pa.table(
        {
            "id": pa.array([1, 2], pa.int64()),
            "st": pa.array([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]),
            "n": pa.array([{"p": {"q": 7}}, {"p": {"q": 8}}]),
        }
    )
    duck.register("l", table)
    assert_same(bt.sql(sql, l=table).collect(), duck.sql(sql))


# --- the f64 aggregates over a DECIMAL column ---------------------------------------


def _decimal_table() -> pa.Table:
    return pa.table(
        {
            "g": pa.array(["a", "a", "b", "b"], pa.string()),
            "o": pa.array([1, 2, 1, 2], pa.int64()),
            "d": pa.array(
                [
                    decimal.Decimal("1.50"),
                    decimal.Decimal("2.50"),
                    decimal.Decimal("3.50"),
                    decimal.Decimal("9.00"),
                ],
                pa.decimal128(7, 2),
            ),
        }
    )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT stddev(d) AS r FROM t",
        "SELECT var_samp(d) AS r FROM t",
        "SELECT median(d) AS r FROM t",
        "SELECT skewness(d) AS r FROM t",
        "SELECT kurtosis(d) AS r FROM t",
        "SELECT g, stddev(d) AS r FROM t GROUP BY g",
        "SELECT g, stddev(d) OVER (PARTITION BY g ORDER BY o) AS r FROM t",
        # The two that must keep their exact decimal accumulation, not be widened away.
        "SELECT sum(d) AS r FROM t",
        "SELECT min(d) AS r FROM t",
        "SELECT max(d) AS r FROM t",
    ],
)
def test_a_decimal_column_reaches_the_f64_aggregates(duck, sql):
    """`STDDEV(price)` over a DECIMAL runs in DuckDB and was refused here.

    The kernels accumulate in `f64` and rejected a decimal outright; the fix widens the
    *input* for exactly the aggregates that already return a DOUBLE, so nothing a decimal
    result could have kept is lost — and `sum`/`min`/`max`, which are exact or
    type-preserving on a decimal, are deliberately left alone.
    """
    table = _decimal_table()
    duck.register("t", table)
    assert_same(bt.sql(sql, t=table).collect(), duck.sql(sql))


def test_sum_over_a_decimal_stays_a_decimal():
    """The widening must not reach `sum`: its decimal accumulation is exact."""
    table = _decimal_table()
    schema = bt.sql("SELECT sum(d) AS r FROM t", t=table).schema
    assert pa.types.is_decimal(schema.field("r").type)
