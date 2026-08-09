"""`IN` must accept every literal `=` accepts, and match the same rows.

`bc_expr::eval::in_list` accelerates `x IN (…)` with typed fast paths keyed on the
column's Arrow type. Each built its member set with `filter_map`, so a literal the arm
could not represent was **silently dropped**:

    date_col IN ('2000-06-30', '2000-09-27')   -- both members filtered away -> no rows

The single-literal spelling was correct throughout, which is what kept this quiet: one
equality is never folded into an `InList`, so `d = '2000-06-30'` matched (the `=` kernel
casts the string to the column's type) while `d IN ('2000-06-30', '2000-09-27')` returned
nothing. TPC-DS q83 returned 0 rows against DuckDB's 24 for exactly this reason.

It was not SQL-only. `Expr.is_in` on the public API had it too, including the purely
numeric `is_in([1.0, 2.0])` against an `Int64` column.

The module's own contract already stated the rule — its untyped fallback delegates to the
OR-of-equality the fold came from, "so `IN` can neither refuse a pair `=` accepts nor
invent one it rejects". The typed arms now honor it: an arm is taken only when it can
represent the whole set, and otherwise falls back to that same `=`-equivalent path.

Every case below is paired with the `=` / `OR` spelling it folds from, because agreement
between the two spellings is the actual invariant — a differential check against DuckDB
alone would not say *which* of the two was wrong.
"""

from __future__ import annotations

import datetime as dt

import duckdb
import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher.plan.expr_ir import col


@pytest.fixture
def tables():
    t = pa.table(
        {
            "d": pa.array(
                [dt.date(2000, 6, 30), dt.date(2000, 9, 27), dt.date(2001, 1, 1), None],
                pa.date32(),
            ),
            "n": pa.array([1, 2, 3, None], pa.int64()),
            "f": pa.array([1.0, 2.0, 3.0, None], pa.float64()),
            "s": pa.array(["a", "b", "c", None]),
        }
    )
    d = duckdb.connect()
    d.register("t", t)
    sess = bt.Session()
    sess.register("t", t)
    return sess, d, t


_CASES = [
    ("date-in-strings", "SELECT d FROM t WHERE d IN ('2000-06-30', '2000-09-27')"),
    ("date-in-typed", "SELECT d FROM t WHERE d IN (DATE '2000-06-30', DATE '2000-09-27')"),
    ("date-in-mixed", "SELECT d FROM t WHERE d IN (DATE '2000-06-30', '2000-09-27')"),
    ("date-or-equals", "SELECT d FROM t WHERE d = '2000-06-30' OR d = '2000-09-27'"),
    ("int-in-floats", "SELECT n FROM t WHERE n IN (1.0, 2.0)"),
    ("int-in-mixed", "SELECT n FROM t WHERE n IN (1, 2.0)"),
    ("float-in-ints", "SELECT f FROM t WHERE f IN (1, 2)"),
    ("int-in-ints", "SELECT n FROM t WHERE n IN (1, 2)"),  # the untouched fast path
    ("str-in-strs", "SELECT s FROM t WHERE s IN ('a', 'b')"),  # the untouched fast path
    ("not-in-strings", "SELECT d FROM t WHERE d NOT IN ('2000-06-30', '2000-09-27')"),
]


@pytest.mark.differential
@pytest.mark.parametrize(("label", "sql"), _CASES, ids=[c[0] for c in _CASES])
def test_in_list_matches_duckdb(tables, label, sql):
    """Membership agrees with the oracle whatever type the literals are written as."""
    sess, duck, _ = tables
    assert_same(sess.sql(sql).collect(), duck.sql(sql))


@pytest.mark.differential
def test_in_agrees_with_the_or_chain_it_folds_from(tables):
    """`IN` and the `OR`-of-equality it is an optimization of must return the same rows.

    This is the invariant the kernel is written against, stated without reference to
    DuckDB: `IN` is a *faster spelling* of the chain, never a different one. It is the
    assertion that localizes a regression to the fold rather than to the comparison.
    """
    sess, _, _ = tables
    in_rows = sess.sql("SELECT d FROM t WHERE d IN ('2000-06-30', '2000-09-27')").collect()
    or_rows = sess.sql("SELECT d FROM t WHERE d = '2000-06-30' OR d = '2000-09-27'").collect()
    assert in_rows.num_rows == or_rows.num_rows == 2, (
        f"IN gave {in_rows.num_rows} rows and the OR chain gave {or_rows.num_rows}; "
        "both must be 2 (the string literals must coerce to the DATE column)"
    )
    assert in_rows.to_pylist() == or_rows.to_pylist()


@pytest.mark.differential
def test_is_in_on_the_public_api_coerces_too(tables):
    """`Expr.is_in` is the same kernel and carries the same fix.

    The SQL cases above all reach `InList` through the optimizer's fold; this one reaches
    it directly from the DataFrame API, which is how a user hits the bug without writing
    any SQL at all.
    """
    _, _, t = tables
    ds = bt.from_arrow(t)
    assert ds.filter(col("d").is_in(["2000-06-30", "2000-09-27"])).collect().num_rows == 2
    assert ds.filter(col("n").is_in([1.0, 2.0])).collect().num_rows == 2
    assert ds.filter(col("d").is_in([dt.date(2000, 6, 30)])).collect().num_rows == 1
    # The already-correct homogeneous paths must be untouched.
    assert ds.filter(col("n").is_in([1, 2])).collect().num_rows == 2
    assert ds.filter(col("s").is_in(["a", "b"])).collect().num_rows == 2
