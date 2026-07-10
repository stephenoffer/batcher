"""Plan-time constant folding of `cast` agrees with DuckDB (and with executing it).

`fold_constants` now evaluates `cast(<literal>, dtype)` at plan time, which is what
collapses the ubiquitous SQL date-interval bound — ``d <= date '1998-12-01' - interval
'90' day`` lowers to ``cast(cast(date, int64) + (-90), date)`` — into a single `Lit`.
A fold that disagreed with the engine would silently change results, so these pin the
folded predicate against DuckDB across the boundary cases: the exact bound and either
side of it, temporal↔integer round trips, and the conversions the fold deliberately
*refuses* (float→int rounding, out-of-range int→date), which the engine still evaluates
and which must still match.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher.kyber.rules.normalize import fold_constants
from batcher.kyber.rules.normalize.fold import _fold_cast
from batcher.plan.expr_ir import Lit
from batcher.plan.logical import Filter
from batcher.plan.visitor import walk
from conftest import assert_same

pytestmark = pytest.mark.differential

# Days spanning the folded 1998-09-02 bound, plus a null.
_DATES = pa.table(
    {
        "d": pa.array(
            [
                dt.date(1998, 9, 1),  # below the bound
                dt.date(1998, 9, 2),  # exactly the bound (inclusive for `<=`)
                dt.date(1998, 9, 3),  # above the bound
                dt.date(1992, 1, 2),
                None,
            ],
            type=pa.date32(),
        ),
        "v": pa.array([1, 2, 3, 4, 5], type=pa.int64()),
    }
)


def _session(duck, table: pa.Table) -> bt.Session:
    duck.register("t", table)
    session = bt.Session()
    session.register("t", table)
    return session


@pytest.mark.parametrize("op", ["<=", "<", ">=", ">", "="])
def test_date_minus_interval_bound_matches_duckdb(duck, op):
    session = _session(duck, _DATES)
    where = f"d {op} date '1998-12-01' - interval '90' day"
    out = session.sql(f"SELECT v FROM t WHERE {where}").collect()
    assert_same(out, duck.sql(f"SELECT v FROM t WHERE {where}"))


def test_date_plus_interval_bound_matches_duckdb(duck):
    session = _session(duck, _DATES)
    where = "d >= date '1992-01-01' + interval '1' day"
    out = session.sql(f"SELECT v FROM t WHERE {where}").collect()
    assert_same(out, duck.sql(f"SELECT v FROM t WHERE {where}"))


def test_interval_bound_folds_to_a_single_literal(duck):
    """The whole cast/add/cast chain collapses to one `Lit`, not merely to fewer nodes."""
    session = _session(duck, _DATES)
    plan = session.sql("SELECT v FROM t WHERE d <= date '1998-12-01' - interval '90' day")._plan
    filters = [n for n in walk(fold_constants(plan)) if isinstance(n, Filter)]
    assert filters, "expected a Filter in the plan"
    rhs = filters[0].predicate.right
    assert isinstance(rhs, Lit), f"predicate right side not folded: {rhs!r}"
    assert rhs.value == dt.date(1998, 9, 2)


def test_temporal_integer_round_trip_is_exact():
    """`date → int64 → date` folds back to the same day (Arrow's epoch-day offset)."""
    day = dt.date(1998, 12, 1)
    as_int = _fold_cast(day, "int64")
    assert as_int is not None
    back = _fold_cast(as_int.value, "date")
    assert back is not None
    assert back.value == day


@pytest.mark.parametrize(
    ("value", "dtype"),
    [
        (1.7, "int64"),  # float→int rounds; the two Arrow implementations may not agree
        ("12", "int64"),  # string→int parses
        (dt.date(1998, 12, 1), "timestamp"),  # temporal→temporal rebases days vs micros
        (10**18, "date"),  # out of date32 range
        (True, "int64"),  # bool is not an integer literal here
    ],
)
def test_unsafe_casts_are_left_to_the_engine(value, dtype):
    """A cast the two Arrow implementations could disagree on must not fold."""
    assert _fold_cast(value, dtype) is None


def test_refused_fold_still_matches_duckdb(duck):
    """A `cast` the folder refuses is executed by the engine, and still agrees."""
    table = pa.table({"x": pa.array([1.4, 1.5, 2.5, -1.5], type=pa.float64())})
    session = _session(duck, table)
    query = "SELECT x FROM t WHERE x > cast(1.7 AS BIGINT)"
    assert_same(session.sql(query).collect(), duck.sql(query))
