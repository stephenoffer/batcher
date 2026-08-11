"""ASOF `tolerance` and `direction="nearest"`, checked against DuckDB and pandas.

Without a tolerance an ASOF join is happy to match a trade against a quote from three days
earlier, because that quote really is the nearest one preceding it. `tolerance` is what
turns "the last value I have" into "the last value that is still current", and getting it
wrong is a silent analytics error rather than a crash — so it is checked against two
independent oracles:

* **DuckDB** has no tolerance clause, but it has `ASOF LEFT JOIN` plus arithmetic, so
  `CASE WHEN left.t - right.t <= tol THEN ... END` expresses exactly the same relation.
* **pandas** `merge_asof` has both `tolerance` and `direction="nearest"` natively, which is
  the only oracle for the nearest search since DuckDB cannot express it.

The edges asserted are the ones a naive implementation gets wrong: the boundary (a distance
*equal* to the tolerance matches), a candidate rejected by tolerance not falling through to
a farther one, ties under `nearest` resolving backward, and tolerance interacting with `by`
groups and with nulls.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.differential

pd = pytest.importorskip("pandas")


def _trades(t, g=None):
    cols = {"t": pa.array(t, type=pa.int64()), "trade": pa.array(list(range(len(t))))}
    if g is not None:
        cols["g"] = pa.array(g, type=pa.string())
    return pa.table(cols)


def _quotes(t, g=None):
    cols = {"t": pa.array(t, type=pa.int64()), "q": pa.array([f"q{i}" for i in range(len(t))])}
    if g is not None:
        cols["g"] = pa.array(g, type=pa.string())
    return pa.table(cols)


def _batcher(left, right, **kw) -> list[str | None]:
    return (
        bt.from_arrow(left)
        .join_asof(bt.from_arrow(right), on="t", **kw)
        .sort("trade")
        .to_pydict()["q"]
    )


def _duck_backward_with_tolerance(duck, left, right, tol: int) -> list[str | None]:
    """DuckDB's ASOF join plus the tolerance the clause itself cannot express."""
    duck.register("l", left)
    duck.register("r", right)
    sql = f"""
        SELECT CASE WHEN l.t - r.t <= {tol} THEN r.q END AS q
        FROM l ASOF LEFT JOIN r ON l.t >= r.t
        ORDER BY l.trade
    """
    return [row[0] for row in duck.sql(sql).fetchall()]


def _pandas(
    left, right, *, direction: str, tolerance=None, by=None, allow_exact_matches=True
) -> list[str | None]:
    lp = left.to_pandas().sort_values("t")
    rp = right.to_pandas().sort_values("t")
    out = pd.merge_asof(
        lp,
        rp,
        on="t",
        by=by,
        direction=direction,
        tolerance=tolerance,
        allow_exact_matches=allow_exact_matches,
        suffixes=("", "_right"),
    ).sort_values("trade")
    return [None if pd.isna(v) else v for v in out["q"].tolist()]


@pytest.mark.parametrize("tol", [0, 1, 2, 3, 5, 100])
def test_tolerance_matches_duckdb_asof_plus_an_explicit_distance_check(duck, tol):
    left = _trades([1, 5, 10, 12, 20])
    right = _quotes([2, 6, 9])
    assert _batcher(left, right, tolerance=tol) == _duck_backward_with_tolerance(
        duck, left, right, tol
    )


@pytest.mark.parametrize("direction", ["backward", "forward", "nearest"])
@pytest.mark.parametrize("tol", [None, 0, 1, 2, 4])
def test_every_direction_and_tolerance_matches_pandas(direction, tol):
    left = _trades([1, 5, 10, 12, 20])
    right = _quotes([2, 6, 9, 15])
    assert _batcher(left, right, direction=direction, tolerance=tol) == _pandas(
        left, right, direction=direction, tolerance=tol
    )


def test_the_tolerance_boundary_is_inclusive():
    """A distance exactly equal to the tolerance matches, as in pandas and Polars."""
    left = _trades([10])
    right = _quotes([7])
    assert _batcher(left, right, tolerance=3) == ["q0"]
    assert _batcher(left, right, tolerance=2) == [None]


def test_a_rejected_candidate_does_not_fall_through_to_a_farther_one():
    """The nearest row is the only candidate: if it fails the tolerance there is no match.

    A search that kept walking would find a *worse* row and call it a match, which is the
    opposite of what a tolerance is for.
    """
    left = _trades([10])
    right = _quotes([1, 2, 3])  # all far away; 3 is the nearest backward
    assert _batcher(left, right, tolerance=5) == [None]


def test_nearest_is_not_pruned_by_the_one_sided_on_bound():
    """`prune_asof_right_by_on_bound` may bound only the end a *directional* join reaches.

    A nearest join reaches both, so the rule must decline. It did not: it read the direction
    as "backward, or else forward", pushed the forward bound onto a nearest join, and deleted
    the very rows the join should have matched — with no error and a plausible answer.
    """
    left = _trades([10])
    right = _quotes([5, 100])  # the nearest is the one *below* the only left key
    assert _batcher(left, right, direction="nearest") == ["q0"]
    assert _batcher(left, right, direction="nearest") == _pandas(left, right, direction="nearest")
    # ...and the plan may not carry a filter that eliminated it.
    plan = (
        bt.from_arrow(left).join_asof(bt.from_arrow(right), on="t", direction="nearest").explain()
    )
    assert "filter" not in plan, plan


def test_nearest_prefers_the_backward_row_on_an_exact_tie():
    left = _trades([10])
    right = _quotes([5, 15])  # both exactly 5 away
    assert _batcher(left, right, direction="nearest") == ["q0"]
    assert _batcher(left, right, direction="nearest") == _pandas(left, right, direction="nearest")


def test_tolerance_applies_within_each_by_group():
    left = _trades([10, 10], g=["a", "b"])
    right = _quotes([9, 1], g=["a", "b"])  # group a is 1 away, group b is 9 away
    got = (
        bt.from_arrow(left)
        .join_asof(bt.from_arrow(right), on="t", by="g", tolerance=2)
        .sort("trade")
        .to_pydict()["q"]
    )
    assert got == ["q0", None]
    assert got == _pandas(left, right, direction="backward", tolerance=2, by="g")


def test_a_duration_tolerance_measures_a_timestamp_key_in_microseconds():
    base = dt.datetime(2024, 1, 1, 12, 0, 0)
    left = pa.table(
        {
            "t": pa.array([base, base + dt.timedelta(minutes=30)]),
            "trade": pa.array([0, 1], type=pa.int64()),
        }
    )
    right = pa.table({"t": pa.array([base - dt.timedelta(minutes=2)]), "q": pa.array(["quote"])})
    # The first trade is 2 minutes after the quote, the second is 32 minutes after.
    assert _batcher(left, right, tolerance="5m") == ["quote", None]
    assert _batcher(left, right, tolerance="1h") == ["quote", "quote"]
    # A `timedelta` is the same tolerance spelled the other way.
    assert _batcher(left, right, tolerance=dt.timedelta(minutes=5)) == ["quote", None]


def test_a_duration_tolerance_measures_a_date_key_in_microseconds_too():
    left = pa.table(
        {
            "t": pa.array([dt.date(2024, 1, 10), dt.date(2024, 2, 1)]),
            "trade": pa.array([0, 1], type=pa.int64()),
        }
    )
    right = pa.table({"t": pa.array([dt.date(2024, 1, 8)]), "q": pa.array(["quote"])})
    assert _batcher(left, right, tolerance="3d") == ["quote", None]


def test_a_float_key_keeps_a_fractional_tolerance():
    left = pa.table({"t": pa.array([1.0, 2.0]), "trade": pa.array([0, 1], type=pa.int64())})
    right = pa.table({"t": pa.array([0.75]), "q": pa.array(["quote"])})
    assert _batcher(left, right, tolerance=0.3) == ["quote", None]
    assert _batcher(left, right, tolerance=0.2) == [None, None]


def test_an_unmeasurable_key_is_rejected_rather_than_silently_ignored():
    """A string `on` orders fine but cannot be subtracted, so a tolerance must error."""
    left = pa.table({"t": pa.array(["b"]), "trade": pa.array([0], type=pa.int64())})
    right = pa.table({"t": pa.array(["a"]), "q": pa.array(["quote"])})
    # Without a tolerance the ordinary ordered search still works.
    assert _batcher(left, right) == ["quote"]
    with pytest.raises(Exception, match="numeric or temporal"):
        _batcher(left, right, tolerance=1)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tolerance": -1}, "non-negative"),
        ({"tolerance": "1mo"}, "calendar unit"),
        ({"tolerance": "nonsense"}, "cannot parse"),
        ({"direction": "sideways"}, "direction must be one of"),
    ],
    ids=["negative", "calendar", "unparseable", "bad_direction"],
)
def test_a_bad_tolerance_or_direction_is_rejected_at_plan_time(kwargs, message):
    left = bt.from_pydict({"t": [1], "trade": [0]})
    right = bt.from_pydict({"t": [1], "q": ["x"]})
    with pytest.raises(PlanError, match=message):
        left.join_asof(right, on="t", **kwargs)


def test_a_null_key_still_matches_nothing_under_a_tolerance():
    left = pa.table({"t": pa.array([None, 10], type=pa.int64()), "trade": pa.array([0, 1])})
    right = _quotes([9])
    assert _batcher(left, right, tolerance=5) == [None, "q0"]


@pytest.mark.parametrize("direction", ["backward", "forward", "nearest"])
def test_tolerance_survives_the_distributed_path(direction):
    """A tolerance honoured single-node and dropped across the shuffle is a wrong answer."""
    n = 500
    left = pa.table(
        {
            "g": pa.array([f"s{i % 9}" for i in range(n)]),
            "t": pa.array([i * 3 for i in range(n)], type=pa.int64()),
            "trade": pa.array(list(range(n)), type=pa.int64()),
        }
    )
    right = pa.table(
        {
            "g": pa.array([f"s{i % 9}" for i in range(n // 2)]),
            "t": pa.array([i * 7 for i in range(n // 2)], type=pa.int64()),
            "q": pa.array([f"q{i}" for i in range(n // 2)]),
        }
    )
    lds, rds = bt.from_arrow(left), bt.from_arrow(right)
    kw = {"on": "t", "by": "g", "direction": direction, "tolerance": 10}
    one = lds.join_asof(rds, **kw).sort("trade").to_pydict()["q"]
    many = lds.repartition(8).join_asof(rds.repartition(4), **kw).sort("trade").to_pydict()["q"]
    assert one == many
    # ...and the tolerance genuinely bites, so the comparison is not two copies of "match
    # everything".
    assert any(v is None for v in one)
    assert any(v is not None for v in one)


def test_strict_matching_ignores_a_right_row_at_the_same_instant():
    """`allow_exact_matches=False` — the guard against look-ahead bias in a backtest.

    A quote stamped at exactly the trade's instant is information the trade did not have.
    Matching it inflates every result downstream and never looks like a bug, which is why
    the strict form exists and why it is checked against pandas rather than asserted alone.
    """
    left = _trades([1, 5, 10])
    right = _quotes([5, 10])
    for direction in ("backward", "forward", "nearest"):
        for exact in (True, False):
            got = _batcher(left, right, direction=direction, allow_exact_matches=exact)
            want = _pandas(left, right, direction=direction, allow_exact_matches=exact)
            assert got == want, f"{direction} allow_exact_matches={exact}"
    # Spelled out for the default direction, so the intent is legible without pandas.
    assert _batcher(left, right) == [None, "q0", "q1"]
    assert _batcher(left, right, allow_exact_matches=False) == [None, None, "q0"]


def test_strict_matching_steps_past_a_whole_run_of_equal_keys():
    """The boundary moves past *every* equal key, not just the last one seen."""
    left = _trades([10])
    right = _quotes([9, 10, 10, 10])
    assert _batcher(left, right) == ["q3"]  # the last equal row
    assert _batcher(left, right, allow_exact_matches=False) == ["q0"]


def test_strict_matching_composes_with_tolerance():
    left = _trades([10])
    right = _quotes([8, 10])
    # Permissive takes the exact row (distance 0); strict falls back to the one 2 away,
    # which a tolerance of 1 then rejects.
    assert _batcher(left, right, tolerance=1) == ["q1"]
    assert _batcher(left, right, tolerance=1, allow_exact_matches=False) == [None]
    assert _batcher(left, right, tolerance=2, allow_exact_matches=False) == ["q0"]
