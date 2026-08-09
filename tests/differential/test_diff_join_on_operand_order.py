"""``JOIN b ON b.k = a.k`` is the same join as ``ON a.k = b.k``.

`from_clause._split_join_on` read the equality's operand *position* to decide which
relation a key belonged to, so an `ON` written right-hand-table-first bound the right
side's column to the left relation. The query then died with

    ColumnNotFoundError: projection '__jk_l0' references unknown column(s) ['bk']

on the plainest form there is — `SELECT * FROM a JOIN b ON b.bk = a.ak`. SQL attaches no
meaning to which side of the `=` a key is typed on, so this rejected valid, ordinary SQL.
It blocked four TPC-DS queries (q72, q75, q78, q93), all of which name the right-hand table
first; q93 writes `store_sales LEFT OUTER JOIN store_returns ON (sr_item_sk = ss_item_sk)`.

Keys are now oriented by which relation owns each column, with the written order preferred
whenever it already resolves — so every join that worked before takes exactly its old path.

These are hard errors rather than silent wrong answers, which is why the cases below assert
against DuckDB rather than probing row counts: if the orientation regresses, the query
raises and the test fails on the exception.
"""

from __future__ import annotations

import duckdb
import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same


@pytest.fixture
def tables():
    a = pa.table({"ak": pa.array([1, 2, 3], pa.int64()), "av": pa.array([10, 20, 30], pa.int64())})
    b = pa.table({"bk": pa.array([1, 2], pa.int64()), "bv": pa.array([10, 200], pa.int64())})
    d = duckdb.connect()
    d.register("a", a)
    d.register("b", b)
    sess = bt.Session()
    sess.register("a", a)
    sess.register("b", b)
    return sess, d


_CASES = [
    ("inner-written-order", "SELECT a.ak, b.bv FROM a JOIN b ON a.ak = b.bk"),
    ("inner-flipped", "SELECT a.ak, b.bv FROM a JOIN b ON b.bk = a.ak"),
    ("left-written-order", "SELECT a.ak, b.bv FROM a LEFT JOIN b ON a.ak = b.bk"),
    ("left-flipped", "SELECT a.ak, b.bv FROM a LEFT JOIN b ON b.bk = a.ak"),
    ("left-flipped-unqualified", "SELECT ak, bv FROM a LEFT JOIN b ON bk = ak"),
    ("right-flipped", "SELECT a.ak, b.bv FROM a RIGHT JOIN b ON b.bk = a.ak"),
    ("full-flipped", "SELECT a.ak, b.bv FROM a FULL JOIN b ON b.bk = a.ak"),
    ("flipped-with-residual", "SELECT a.ak, b.bv FROM a JOIN b ON b.bk = a.ak AND b.bv > 100"),
    # One key each way in the same ON: orientation must be decided per conjunct.
    ("mixed-order-two-keys", "SELECT a.ak FROM a JOIN b ON b.bk = a.ak AND a.av = b.bv"),
    # Both sides own a column of that name, so membership cannot decide and the written
    # order must be kept — the self-join path `_disambiguate_columns` renames apart.
    ("same-name-keys", "SELECT x.ak FROM a x JOIN a y ON y.ak = x.ak"),
]


@pytest.mark.differential
@pytest.mark.parametrize(("label", "sql"), _CASES, ids=[c[0] for c in _CASES])
def test_on_clause_operand_order_does_not_change_the_join(tables, label, sql):
    """Either operand order gives DuckDB's answer, for every join type."""
    sess, duck = tables
    assert_same(sess.sql(sql).collect(), duck.sql(sql))


@pytest.mark.differential
def test_flipped_and_written_order_agree_with_each_other(tables):
    """The two spellings of one join return the same rows.

    Stated independently of the oracle: whatever the engine computes for `a.ak = b.bk`, it
    must compute for `b.bk = a.ak`. A regression that broke both spellings the same way
    would still pass this, which is why the DuckDB comparison above is the primary check —
    this one localizes the failure to the orientation when it does fire.
    """
    sess, _ = tables
    written = sess.sql("SELECT a.ak, b.bv FROM a LEFT JOIN b ON a.ak = b.bk").collect()
    flipped = sess.sql("SELECT a.ak, b.bv FROM a LEFT JOIN b ON b.bk = a.ak").collect()
    assert written.num_rows == flipped.num_rows
    assert sorted(written.to_pylist(), key=repr) == sorted(flipped.to_pylist(), key=repr)
