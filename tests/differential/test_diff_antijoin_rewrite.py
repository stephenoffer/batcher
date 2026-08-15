"""`LEFT JOIN ... WHERE right_key IS NULL` against DuckDB — the anti-join rewrite.

`kyber.rules.joins.rewrites.left_join_null_key_to_antijoin` turns that shape into an anti
join, which is exact only because an equi-join never matches a null key. These cases are the
boundary of that argument: a null key on the build side, an `IS NULL` on a right *payload*
column (where the rewrite must decline), and the shapes that decide whether the right-hand
columns may be dropped at all.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same


@pytest.fixture
def tables(duck):
    left = pa.table(
        {
            "tid": [1, 2, 3, 4, 5, 6, None],
            "item": [10, 20, 30, 40, 50, 60, 70],
            "qty": [1, 2, 3, 4, 5, 6, 7],
        }
    )
    # Row (2, 20) matches; (4, None) has a null key, so it can match nothing; (99, 990) is
    # a right-only row; (6, 60) matches but carries a null payload.
    right = pa.table(
        {
            "rtid": [2, 4, 99, 6],
            "ritem": [20, None, 990, 60],
            "amt": [9.0, 8.0, 7.0, None],
        }
    )
    duck.register("l", left)
    duck.register("r", right)
    return left, right


def _session(left, right):
    sess = bt.Session()
    sess.register("l", left)
    sess.register("r", right)
    return sess


def test_null_key_filter_matches_duckdb(duck, tables):
    """The rewrite's own shape, aggregated — TPC-DS q78 writes it exactly this way."""
    left, right = tables
    sql = """SELECT tid, sum(qty) AS q FROM l LEFT JOIN r ON tid = rtid AND item = ritem
             WHERE rtid IS NULL GROUP BY tid"""
    assert_same(_session(left, right).sql(sql).collect(), duck.sql(sql))


def test_null_key_filter_projected_matches_duckdb(duck, tables):
    """The same predicate under a plain projection rather than an aggregate."""
    left, right = tables
    sql = (
        "SELECT tid, item, qty FROM l LEFT JOIN r ON tid = rtid AND item = ritem WHERE rtid IS NULL"
    )
    assert_same(_session(left, right).sql(sql).collect(), duck.sql(sql))


def test_a_null_key_on_the_build_side_still_matches_duckdb(duck, tables):
    """The case the rewrite's correctness argument turns on.

    `r` holds a row whose `ritem` is null. It can never match — SQL equality against null is
    unknown — so every left row stays unmatched on it, which is what makes "null right key
    after the join" mean "no match" and the anti join exact. Joining on that column alone is
    what exercises it.
    """
    left, right = tables
    sql = "SELECT tid, item FROM l LEFT JOIN r ON item = ritem WHERE ritem IS NULL"
    assert_same(_session(left, right).sql(sql).collect(), duck.sql(sql))


def test_is_null_on_a_payload_column_is_not_an_antijoin(duck, tables):
    """`amt IS NULL` keeps matched rows whose payload happens to be null.

    Row (6, 60) matches and carries a null `amt`, so an anti join would wrongly drop it. The
    rewrite must decline here, and the answer must still be DuckDB's.
    """
    left, right = tables
    sql = "SELECT tid, item FROM l LEFT JOIN r ON tid = rtid AND item = ritem WHERE amt IS NULL"
    assert_same(_session(left, right).sql(sql).collect(), duck.sql(sql))


def test_extra_predicates_on_the_left_survive(duck, tables):
    left, right = tables
    sql = """SELECT tid, item FROM l LEFT JOIN r ON tid = rtid AND item = ritem
             WHERE rtid IS NULL AND qty > 2"""
    assert_same(_session(left, right).sql(sql).collect(), duck.sql(sql))


def test_a_predicate_reading_a_right_column_blocks_the_rewrite(duck, tables):
    """A surviving reference to a right column means the columns cannot be dropped."""
    left, right = tables
    sql = """SELECT tid FROM l LEFT JOIN r ON tid = rtid AND item = ritem
             WHERE rtid IS NULL AND (amt IS NULL OR amt > 0)"""
    assert_same(_session(left, right).sql(sql).collect(), duck.sql(sql))


def test_selecting_a_right_column_blocks_the_rewrite(duck, tables):
    left, right = tables
    sql = "SELECT tid, amt FROM l LEFT JOIN r ON tid = rtid AND item = ritem WHERE rtid IS NULL"
    assert_same(_session(left, right).sql(sql).collect(), duck.sql(sql))


def test_an_empty_right_side_keeps_every_left_row(duck):
    left = pa.table({"tid": [1, 2, 3], "qty": [1, 2, 3]})
    right = pa.table({"rtid": pa.array([], pa.int64()), "amt": pa.array([], pa.float64())})
    duck.register("l", left)
    duck.register("r", right)
    sql = "SELECT tid, qty FROM l LEFT JOIN r ON tid = rtid WHERE rtid IS NULL"
    assert_same(_session(left, right).sql(sql).collect(), duck.sql(sql))


def test_an_empty_left_side_produces_nothing(duck):
    left = pa.table({"tid": pa.array([], pa.int64()), "qty": pa.array([], pa.int64())})
    right = pa.table({"rtid": [1, 2], "amt": [1.0, 2.0]})
    duck.register("l", left)
    duck.register("r", right)
    sql = "SELECT tid, qty FROM l LEFT JOIN r ON tid = rtid WHERE rtid IS NULL"
    assert_same(_session(left, right).sql(sql).collect(), duck.sql(sql))


def test_a_duplicated_build_key_does_not_change_the_answer(duck):
    """An anti join tests membership, so build-side duplicates must not fan out or drop rows."""
    left = pa.table({"tid": [1, 2, 3, 4], "qty": [1, 2, 3, 4]})
    right = pa.table({"rtid": [2, 2, 2, 3], "amt": [1.0, 2.0, 3.0, 4.0]})
    duck.register("l", left)
    duck.register("r", right)
    sql = "SELECT tid, qty FROM l LEFT JOIN r ON tid = rtid WHERE rtid IS NULL"
    assert_same(_session(left, right).sql(sql).collect(), duck.sql(sql))


def test_the_rewrite_survives_multiple_morsels(duck):
    """More rows than one morsel holds, so the anti join runs across the parallel path."""
    n = 60_000
    left = pa.table({"tid": list(range(n)), "qty": [i % 7 for i in range(n)]})
    right = pa.table({"rtid": list(range(0, n, 3)), "amt": [1.0] * len(range(0, n, 3))})
    duck.register("l", left)
    duck.register("r", right)
    sql = "SELECT sum(qty) AS s, count(*) AS c FROM l LEFT JOIN r ON tid = rtid WHERE rtid IS NULL"
    assert_same(_session(left, right).sql(sql).collect(), duck.sql(sql))


def test_streamed_batches_agree_with_collect(duck):
    """`iter_batches` is a different executor path from `collect` on the same rewritten plan."""
    n = 40_000
    left = pa.table({"tid": list(range(n)), "qty": [i % 5 for i in range(n)]})
    right = pa.table({"rtid": list(range(0, n, 4)), "amt": [2.0] * len(range(0, n, 4))})
    duck.register("l", left)
    duck.register("r", right)
    sql = "SELECT tid, qty FROM l LEFT JOIN r ON tid = rtid WHERE rtid IS NULL"
    ds = _session(left, right).sql(sql)
    batches = list(ds.iter_batches())
    out = pa.Table.from_batches(batches) if batches else ds.collect()
    assert_same(out, duck.sql(sql))
