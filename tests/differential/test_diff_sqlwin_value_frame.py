"""SQL value-function window framing vs DuckDB (ledger B90).

`last_value`/`nth_value OVER (ORDER BY …)` must run under SQL's default frame
(`RANGE UNBOUNDED PRECEDING TO CURRENT ROW`) — the *running* value, not the
whole-partition last. `first_value` is frame-independent. Explicit ROWS/RANGE
frames on these value functions must also match DuckDB.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from conftest import assert_same

pytestmark = pytest.mark.differential


@pytest.fixture
def t():
    return pa.table(
        {
            "i": [1, 2, 3, 4, 5, 6],
            "g": ["a", "a", "a", "b", "b", "b"],
            # A tied order key so RANGE peer semantics are exercised.
            "k": [10, 10, 20, 30, 30, 30],
            "v": [10, 20, 30, 40, 50, 60],
        }
    )


def _same(duck, t, sql):
    duck.register("t", t)
    assert_same(bt.sql(sql, t=t).collect(), duck.sql(sql))


def test_last_value_default_frame_running(duck, t):
    # DuckDB: [10, 20, 30, ...] — the current row's value, not [30,30,30,...].
    _same(duck, t, "SELECT i, last_value(v) OVER (ORDER BY i) AS l FROM t")


def test_last_value_default_frame_with_ties(duck, t):
    # RANGE default over a tied key: last_value is the last peer of the current row.
    _same(duck, t, "SELECT i, last_value(v) OVER (ORDER BY k) AS l FROM t")


def test_last_value_partitioned_running(duck, t):
    _same(
        duck,
        t,
        "SELECT i, last_value(v) OVER (PARTITION BY g ORDER BY i) AS l FROM t",
    )


def test_first_value_default_frame(duck, t):
    _same(duck, t, "SELECT i, first_value(v) OVER (ORDER BY i) AS f FROM t")


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_nth_value_default_frame(duck, t, n):
    # nth_value is null until the running frame reaches the nth row, then that value.
    _same(duck, t, f"SELECT i, nth_value(v, {n}) OVER (ORDER BY i) AS nv FROM t")


def test_nth_value_partitioned_with_ties(duck, t):
    _same(
        duck,
        t,
        "SELECT i, nth_value(v, 2) OVER (PARTITION BY g ORDER BY k) AS nv FROM t",
    )


@pytest.mark.parametrize(
    "frame",
    [
        "ROWS BETWEEN 1 PRECEDING AND CURRENT ROW",
        "ROWS BETWEEN 2 PRECEDING AND CURRENT ROW",
        "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW",
        "ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING",
    ],
)
def test_last_value_explicit_rows_frame(duck, t, frame):
    _same(
        duck,
        t,
        f"SELECT i, last_value(v) OVER (PARTITION BY g ORDER BY i {frame}) AS l FROM t",
    )


@pytest.mark.parametrize(
    "frame",
    [
        "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW",
        "ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING",
    ],
)
def test_first_value_explicit_rows_frame(duck, t, frame):
    _same(
        duck,
        t,
        f"SELECT i, first_value(v) OVER (ORDER BY i {frame}) AS f FROM t",
    )


def test_nth_value_explicit_rows_frame(duck, t):
    _same(
        duck,
        t,
        "SELECT i, nth_value(v, 2) OVER (ORDER BY i ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) "
        "AS nv FROM t",
    )


@pytest.mark.parametrize("fn", ["first_value", "last_value"])
def test_value_single_bound_frame(duck, t, fn):
    # `ROWS 1 PRECEDING` == `ROWS BETWEEN 1 PRECEDING AND CURRENT ROW`; the end must
    # default to CURRENT ROW, not UNBOUNDED FOLLOWING.
    _same(
        duck,
        t,
        f"SELECT i, {fn}(v) OVER (PARTITION BY g ORDER BY i ROWS 1 PRECEDING) AS x FROM t",
    )
