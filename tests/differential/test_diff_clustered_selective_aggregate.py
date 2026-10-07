"""A grouped aggregate over a *clustered* selective join, against DuckDB.

The streaming executor deals a root aggregate's input to its workers in interleaved
morsel-sized pieces (`stream::parallel::interleaved_shards`) so that a fact table stored in
the order a join selects on -- TPC-DS `inventory` by date -- does not hand all its surviving
rows to one worker. The fixture is that shape: a date-sorted fact table large enough to be
cut into many more pieces than workers (2M rows is ~120 morsels), joined to a dimension
keeping one narrow date range, with NULL keys, NULL values, NaN and -0.0 in the aggregated
columns, and order-insensitive aggregates of every kind the interleaved path accepts. An
order-sensitive aggregate (`arg_min`) and an ordered result are checked too, because those
must keep the contiguous cut.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_same_ordered

_ROWS = 2_000_000


@pytest.fixture(scope="module")
def tables():
    rng = np.random.default_rng(7)
    date = np.sort(rng.integers(0, 400, _ROWS))
    item = rng.integers(0, 5_000, _ROWS)
    qty = rng.integers(0, 1_000, _ROWS).astype(float)
    qty[rng.integers(0, _ROWS, 500)] = np.nan
    qty[rng.integers(0, _ROWS, 500)] = -0.0
    qty_null = rng.random(_ROWS) < 0.01
    item_null = rng.random(_ROWS) < 0.001
    fact = pa.table(
        {
            "d": pa.array(date),
            "item": pa.array(item, mask=item_null),
            "q": pa.array(qty, mask=qty_null),
            "n": pa.array(rng.integers(0, 50, _ROWS)),
        }
    )
    dim = pa.table({"d_sk": pa.array(np.arange(400)), "moy": pa.array(np.arange(400) // 30)})
    return fact, dim


@pytest.fixture
def sess(duck, tables):
    fact, dim = tables
    duck.register("fact", fact)
    duck.register("dim", dim)
    s = bt.Session()
    s.register("fact", fact)
    s.register("dim", dim)
    return s


def test_grouped_order_insensitive_aggregates(duck, sess):
    sql = """
        SELECT item, moy, count(*) AS c, count(q) AS cq, sum(n) AS sn, avg(q) AS aq,
               min(q) AS lo, max(q) AS hi, stddev_samp(n) AS sd
        FROM fact JOIN dim ON d = d_sk
        WHERE moy IN (3, 4)
        GROUP BY item, moy
    """
    assert_same(sess.sql(sql).collect(), duck.sql(sql))


def test_global_aggregate(duck, sess):
    sql = """
        SELECT count(*) AS c, sum(n) AS sn, count(DISTINCT item) AS items, max(q) AS hi
        FROM fact JOIN dim ON d = d_sk WHERE moy = 7
    """
    assert_same(sess.sql(sql).collect(), duck.sql(sql))


def test_empty_selection(duck, sess):
    sql = """
        SELECT item, sum(n) AS sn FROM fact JOIN dim ON d = d_sk
        WHERE moy = 99 GROUP BY item
    """
    assert_same(sess.sql(sql).collect(), duck.sql(sql))


def test_order_sensitive_aggregate_keeps_the_contiguous_cut(duck, sess):
    """`arg_min` names *which* row, so its shards stay contiguous.

    Ties on `n` make the chosen item engine-specific, so what is checked is that it is a
    valid choice: some row of that month carries that item at the minimum.
    """
    sql = """
        SELECT moy, arg_min(item, n) AS first_item, min(n) AS lo
        FROM fact JOIN dim ON d = d_sk WHERE moy BETWEEN 2 AND 5 AND item IS NOT NULL
        GROUP BY moy
    """
    got = sess.sql(sql).collect().to_pydict()
    want = duck.sql(sql).to_arrow_table().to_pydict()
    assert sorted(zip(got["moy"], got["lo"], strict=True)) == sorted(
        zip(want["moy"], want["lo"], strict=True)
    )
    for moy, item, lo in zip(got["moy"], got["first_item"], got["lo"], strict=True):
        hits = duck.sql(
            f"SELECT count(*) FROM fact JOIN dim ON d = d_sk "
            f"WHERE moy = {moy} AND item = {item} AND n = {lo}"
        ).fetchone()[0]
        assert hits > 0, f"arg_min picked item {item} for month {moy}, which has no n = {lo}"


def test_sorted_result_in_order(duck, sess):
    """An ordered result read in order, never with the order-independent `assert_same`."""
    sql = """
        SELECT item, sum(n) AS sn FROM fact JOIN dim ON d = d_sk
        WHERE moy = 5 AND item IS NOT NULL GROUP BY item ORDER BY sn DESC, item LIMIT 50
    """
    assert_same_ordered(sess.sql(sql).collect(), duck.sql(sql))
