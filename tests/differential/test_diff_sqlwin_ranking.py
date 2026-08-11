"""SQL ranking/value window functions newly wired in the SQL translator, vs DuckDB.

`percent_rank`/`cume_dist`/`ntile` (ranking family) and `nth_value` were previously
`NotImplementedError` in the SQL parser though the runtime supports them, as was
`lag`/`lead` with a default value — which is now rewritten onto the functions the engine
does have rather than refused.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential


@pytest.fixture
def t():
    return pa.table(
        {
            "i": [1, 2, 3, 4, 5, 6, 7],
            "g": ["a", "a", "a", "a", "b", "b", "b"],
            "v": [10, 10, 20, 30, 40, 50, 50],
        }
    )


def _same(duck, t, sql):
    duck.register("t", t)
    assert_same(bt.sql(sql, t=t).collect(), duck.sql(sql))


def test_percent_rank(duck, t):
    _same(duck, t, "SELECT i, percent_rank() OVER (PARTITION BY g ORDER BY v) AS pr FROM t")


def test_cume_dist(duck, t):
    _same(duck, t, "SELECT i, cume_dist() OVER (PARTITION BY g ORDER BY v) AS cd FROM t")


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_ntile(duck, t, n):
    _same(duck, t, f"SELECT i, ntile({n}) OVER (PARTITION BY g ORDER BY i) AS q FROM t")


def test_ntile_no_partition(duck, t):
    _same(duck, t, "SELECT i, ntile(3) OVER (ORDER BY i) AS q FROM t")


def test_all_ranking_in_one_select(duck, t):
    _same(
        duck,
        t,
        "SELECT i, "
        "row_number() OVER (ORDER BY i) AS rn, "
        "rank() OVER (ORDER BY v) AS rk, "
        "dense_rank() OVER (ORDER BY v) AS dr, "
        "percent_rank() OVER (ORDER BY v) AS pr, "
        "cume_dist() OVER (ORDER BY v) AS cd, "
        "ntile(2) OVER (ORDER BY i) AS q "
        "FROM t",
    )


def test_lag_lead_offset_and_negative(duck, t):
    _same(
        duck,
        t,
        "SELECT i, lag(v, 2) OVER (ORDER BY i) AS lg, lead(v, 1) OVER (ORDER BY i) AS ld FROM t",
    )


def test_lag_with_default_value_fills_the_out_of_range_rows(duck, t):
    # This used to assert a *rejection*: the window operator has no default parameter, so
    # the translator refused the three-argument form rather than return NULL where SQL
    # returns the default. The form is now rewritten into window functions the engine does
    # have (`CASE WHEN row_number() OVER w <= n THEN d ELSE lag(x, n) END`), so the
    # assertion is the answer itself — strictly stronger than the refusal it replaces.
    # `tests/differential/test_diff_sql_lag_lead_default.py` covers the rewrite in full,
    # including the NULL-inside-the-partition case a COALESCE would get wrong.
    _same(duck, t, "SELECT i, lag(v, 1, -1) OVER (ORDER BY i) AS l FROM t")
