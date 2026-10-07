"""Window counts, grouped QUALIFY, several DISTINCT arguments and UDFs in grouped SQL vs DuckDB.

* ``bt.count().over(keys)`` raised "unknown window function 'count_star'" although SQL's
  ``count(*) OVER`` worked. It now counts rows, nulls included, with a null key forming
  its own partition: the "keep the rows of groups with at least N rows" filter.
* ``QUALIFY`` beside ``GROUP BY`` was refused. SQL evaluates it after the grouping, the
  HAVING and the windows, over the grouped rows.
* ``SUM(DISTINCT a), AVG(DISTINCT b), COUNT(*)`` -- two different DISTINCT arguments --
  was refused; it is answered by Spark's Expand rewrite.
* A registered Python function anywhere in an aggregate or window query was refused. It
  now runs before the pass when it reads input rows (a key, an aggregate's argument, a
  window's partition) and after it when it reads the pass's results. DuckDB is given the
  same function through ``create_function``.
"""

from __future__ import annotations

import dataclasses

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import batcher as bt
from _harness import assert_same
from batcher.config import active_config, set_config

pytestmark = pytest.mark.differential

_T = pa.table(
    {
        "k": pa.array([1, 1, 1, 2, 2, None, None, 3, 4, 4], pa.int64()),
        "x": pa.array([1, 2, 2, None, 5, 3, 3, 7, None, None], pa.int64()),
        "y": pa.array([1, 1, 3, 4, 4, None, 4, 1, 9, 9], pa.int64()),
    }
)


def _times_ten(v):
    return None if v is None else int(v) * 10


def _plus_one(v):
    return None if v is None else int(v) + 1


@pytest.fixture
def session(duck):
    duck.register("t", _T)
    for name, fn in (("times_ten", _times_ten), ("plus_one", _plus_one)):
        duck.create_function(name, fn, ["HUGEINT"], "BIGINT", null_handling="special")
    s = bt.Session()
    s.register("t", _T)
    s.register_function("times_ten", lambda v: pc.multiply(v, 10), result_type="int64")
    s.register_function("plus_one", lambda v: pc.add(v, 1), result_type="int64")
    return s


def _check(session, duck, q):
    got = session.sql(q).collect()
    assert_same(got, duck.sql(q))
    return got


def test_window_count_matches_count_star_over(duck):
    duck.register("t", _T)
    got = bt.from_arrow(_T).with_columns(n=bt.count().over("k"))
    assert_same(got.collect(), duck.sql("SELECT *, count(*) OVER (PARTITION BY k) AS n FROM t"))


def test_window_count_keeps_the_rows_of_big_groups(duck):
    duck.register("t", _T)
    got = bt.from_arrow(_T).filter(bt.count().over("k") >= 2)
    want = duck.sql("SELECT * FROM t QUALIFY count(*) OVER (PARTITION BY k) >= 2")
    assert_same(got.collect(), want)
    empty = bt.from_arrow(_T).filter(bt.col("x") > 100).filter(bt.count().over("k") >= 2)
    assert empty.collect().num_rows == 0


@pytest.mark.parametrize("streaming", [True, False])
def test_window_count_on_a_sharded_input(duck, streaming):
    """Above the 65,536-row sharding threshold, on both executors."""
    n = 100_000
    big = pa.table({"k": pa.array([None if i % 7 == 0 else i % 5 for i in range(n)], pa.int64())})
    duck.register("big", big)
    prev = active_config()
    set_config(prev.replace(execution=dataclasses.replace(prev.execution, streaming=streaming)))
    try:
        got = bt.from_arrow(big).filter(bt.count().over("k") > 17_000).collect()
    finally:
        set_config(prev)
    want = duck.sql("SELECT * FROM big QUALIFY count(*) OVER (PARTITION BY k) > 17000")
    assert 0 < got.num_rows < n
    assert_same(got, want)


@pytest.mark.parametrize(
    "q",
    [
        "SELECT k, sum(x) AS s FROM t GROUP BY k "
        "QUALIFY row_number() OVER (ORDER BY sum(x) DESC NULLS LAST, k) = 1",
        "SELECT k, sum(x) AS total, rank() OVER (ORDER BY k NULLS FIRST) AS r FROM t GROUP BY k "
        "QUALIFY r <= 2",
        "SELECT k, sum(x) AS total FROM t GROUP BY k "
        "QUALIFY rank() OVER (ORDER BY total DESC NULLS LAST) <= 2",
        "SELECT k, count(*) AS c FROM t GROUP BY k HAVING count(*) > 1 "
        "QUALIFY rank() OVER (ORDER BY k NULLS FIRST) = 1",
        "SELECT k, count(*) AS c FROM t GROUP BY k QUALIFY sum(count(*)) OVER () > 3",
        "SELECT k * 10 AS kk, sum(x) AS total, rank() OVER (ORDER BY k) AS r FROM t GROUP BY k "
        "QUALIFY kk > 10",
        "SELECT k, sum(x) AS s FROM t WHERE x > 100 GROUP BY k "
        "QUALIFY row_number() OVER (ORDER BY k) = 1",
        "SELECT k, x FROM t QUALIFY count(*) OVER (PARTITION BY k) >= 2",
    ],
)
def test_grouped_qualify(session, duck, q):
    _check(session, duck, q)


@pytest.mark.parametrize(
    "q",
    [
        "SELECT k, sum(DISTINCT x) AS a, avg(DISTINCT y) AS b, count(*) AS c FROM t GROUP BY k",
        "SELECT sum(DISTINCT x) AS a, avg(DISTINCT y) AS b, count(*) AS c FROM t",
        "SELECT sum(DISTINCT x) AS a, avg(DISTINCT y) AS b, count(*) AS c FROM t WHERE x > 100",
        "SELECT k, sum(DISTINCT x) AS a, avg(DISTINCT y) AS b, count(*) AS c, min(x) AS d "
        "FROM t GROUP BY k",
        "SELECT k, count(DISTINCT x) AS a, sum(DISTINCT y) AS b, avg(x) AS c FROM t GROUP BY k",
        "SELECT k, sum(DISTINCT x) AS a, median(y) AS b FROM t GROUP BY k",
        "SELECT k, sum(DISTINCT x) AS a, sum(DISTINCT y) AS b, sum(DISTINCT x + y) AS c "
        "FROM t GROUP BY k",
    ],
)
def test_several_distinct_arguments(session, duck, q):
    _check(session, duck, q)


@pytest.mark.parametrize(
    "q",
    [
        "SELECT times_ten(k) AS a, sum(x) AS s FROM t GROUP BY times_ten(k)",
        "SELECT k, sum(times_ten(x)) AS s FROM t GROUP BY k",
        "SELECT k, times_ten(sum(x)) AS s FROM t GROUP BY k",
        "SELECT times_ten(k) AS a, count(*) AS c FROM t GROUP BY k",
        "SELECT plus_one(times_ten(k)) AS a, count(*) AS c FROM t GROUP BY 1",
        "SELECT times_ten(sum(x)) + 1 AS s, plus_one(count(*)) AS c FROM t",
        "SELECT k, sum(x) AS s FROM t GROUP BY k HAVING sum(times_ten(x)) > 20",
        "SELECT k, times_ten(count(DISTINCT x)) AS a, sum(DISTINCT times_ten(y)) AS b "
        "FROM t GROUP BY k",
        "SELECT k, x, sum(times_ten(x)) OVER (PARTITION BY k) AS s FROM t",
        "SELECT k, x, times_ten(sum(x) OVER (PARTITION BY k)) AS s FROM t",
        "SELECT k, x, y, rank() OVER (PARTITION BY times_ten(k) ORDER BY plus_one(x), y) AS r "
        "FROM t",
        "SELECT k, times_ten(sum(x)) AS s FROM t WHERE x > 100 GROUP BY k",
    ],
)
def test_registered_function_in_aggregate_and_window_queries(session, duck, q):
    _check(session, duck, q)


def test_registered_function_items_keep_their_written_names(session):
    """An unaliased item is named after the call as written, not the internal column."""
    grouped = session.sql("SELECT times_ten(k), count(*) AS c FROM t GROUP BY times_ten(k)")
    assert grouped.columns == ["times_ten(k)", "c"]
    plain = session.sql("SELECT times_ten(x) FROM t")
    assert plain.columns == ["times_ten(x)"]


def test_registered_function_over_a_grouped_value_outside_select_is_refused(session):
    with pytest.raises(bt.PlanError, match="supported only in the SELECT list"):
        session.sql("SELECT k, sum(x) FROM t GROUP BY k HAVING times_ten(k) > 10").collect()
