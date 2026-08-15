"""A text predicate is *evaluated* against the values that were measured, not guessed at.

`contains`/`starts_with`/`LIKE` used a single prior for every column. The columns those
predicates actually filter — `status`, `country`, `category`, `event_type` — hold a handful
of values, all of them in the most-common-value table, so the predicate can simply be run
against them. The estimate then splits into a measured part and a guessed one:

    Σ f(v) over the listed values the pattern matches  +  residual_mass · prior

which is exact on the covered mass and degrades smoothly to the old prior on a
high-cardinality free-text column, where the table covers little.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.config import active_config
from batcher.kyber.stats import StatsEstimator
from batcher.kyber.stats.selectivity import predicate_selectivity as sel
from batcher.kyber.stats.selectivity.patterns import _like_matches

pytestmark = pytest.mark.unit

_CFG = active_config().optimizer.cardinality
# A covering table: five values, all of the mass. Four of the five contain "a".
_COVERING = {"w": {"alpha": 0.2, "beta": 0.2, "gamma": 0.2, "delta": 0.2, "epsilon": 0.2}}
_NDV = {"w": 5.0}


def _s(expr, mcv=None, nulls=None):
    return sel(expr, _NDV, _CFG, None, mcv, None, nulls)


@pytest.mark.parametrize(
    ("value", "pattern", "expected"),
    [
        ("alpha", "a%", True),
        ("alpha", "%a", True),
        ("alpha", "%ph%", True),
        ("alpha", "alpha", True),
        ("alpha", "alph", False),
        ("alpha", "al_ha", True),
        ("alpha", "%z%", False),
        ("", "%", True),
        ("", "", True),
        ("abc", "___", True),
        ("abc", "__", False),
        # The shape that makes a regex translation of `LIKE` blow up: many `%` runs and a
        # final mismatch. The two-pointer matcher answers it without backtracking.
        ("a" * 40, "%a%a%a%a%a%b", False),
    ],
)
def test_like_matching_follows_sql_wildcard_semantics(value, pattern, expected):
    assert _like_matches(value, pattern) is expected


def test_contains_is_exact_when_the_table_covers_the_column():
    # alpha, beta, gamma, delta contain "a"; epsilon does not.
    assert _s(bt.col("w").str.contains("a"), _COVERING) == pytest.approx(0.8)
    assert _s(bt.col("w").str.contains("z"), _COVERING) == pytest.approx(0.0)
    assert _s(bt.col("w").str.contains(""), _COVERING) == pytest.approx(1.0)


def test_anchored_predicates_are_exact_when_the_table_covers_the_column():
    assert _s(bt.col("w").str.starts_with("a"), _COVERING) == pytest.approx(0.2)
    assert _s(bt.col("w").str.ends_with("a"), _COVERING) == pytest.approx(0.8)


def test_like_is_exact_when_the_table_covers_the_column():
    assert _s(bt.col("w").str.contains("a"), _COVERING) == pytest.approx(0.8)
    assert _s(bt.col("w").str.like("%a"), _COVERING) == pytest.approx(0.8)
    assert _s(bt.col("w").str.like("a%"), _COVERING) == pytest.approx(0.2)
    assert _s(bt.col("w").str.like("_elta"), _COVERING) == pytest.approx(0.2)


def test_the_prior_applies_only_to_the_uncovered_mass():
    """A partial table: what it lists is measured, the rest keeps the prior."""
    partial = {"w": {"alpha": 0.1, "epsilon": 0.1}}  # 20% covered, 80% residual
    got = _s(bt.col("w").str.contains("a"), partial)
    assert got == pytest.approx(0.1 + 0.8 * _CFG.substring_selectivity)


def test_no_table_keeps_the_prior_exactly():
    assert _s(bt.col("w").str.contains("a")) == pytest.approx(_CFG.substring_selectivity)
    assert _s(bt.col("w").str.contains("a"), {}) == pytest.approx(_CFG.substring_selectivity)


def test_a_regex_is_never_matched_against_the_measured_values():
    """The engine matches with Rust's linear-time `regex`; Python's `re` backtracks.

    A pattern that is well-behaved in the engine can be catastrophic here, and it would be
    catastrophic *in the planner*. So a regex keeps its pattern-shape prior even when the
    table would decide it.
    """
    got = _s(bt.col("w").str.regexp_matches("^alpha$"), _COVERING)
    # The exact-equality shape reads the frequency of the literal, not a match count.
    assert got == pytest.approx(0.2)
    floating = _s(bt.col("w").str.regexp_matches("a+"), _COVERING)
    assert floating == pytest.approx(_CFG.substring_selectivity)


def test_the_null_rows_are_not_handed_to_the_pattern():
    """Only the covered mass is measured; a null row matches nothing."""
    table = {"w": {"alpha": 0.4, "epsilon": 0.3}}  # 70% covered, 30% null, 0% residual
    got = _s(bt.col("w").str.contains("a"), table, nulls={"w": 0.3})
    assert got == pytest.approx(0.4)


def test_the_estimate_tracks_the_executed_row_count():
    words = ["alpha", "beta", "gamma", "delta", "epsilon"]
    rows = 5000
    values = [words[i % 5] for i in range(rows)]
    learned = {"__column_mcv__": {"w": {w: values.count(w) / rows for w in words}}}
    dataset = bt.from_pydict({"w": values}).filter(bt.col("w").str.contains("a"))
    stats = [s.statistics() for s in dataset._sources]
    estimated = (
        StatsEstimator(dataset._sources, learned, _CFG, source_stats=stats)
        .estimate(dataset._plan)
        .rows
    )
    executed = dataset.count()
    assert executed == rows * 4 // 5
    assert estimated == pytest.approx(executed, rel=0.01)


# --- statistics are read through a value-preserving cast --------------------------


def test_a_cast_does_not_hide_the_column_from_the_estimator():
    """`cast(x AS DOUBLE) = 5.0` is a comparison on `x`, and `x`'s statistics describe it.

    Without this the comparison had no column at all and took the flat cold-start constant.
    """
    ndv = {"x": 100.0}
    plain = sel(bt.col("x") == 5, ndv, _CFG)
    casted = sel(bt.col("x").cast("float64") == 5.0, ndv, _CFG)
    assert casted == pytest.approx(plain)
    assert casted == pytest.approx(0.01)


def test_a_cast_does_not_hide_the_column_from_a_range_or_an_in_list():
    ndv = {"x": 100.0}
    bounds = {"x": (0, 99)}
    plain = sel(bt.col("x") < 50, ndv, _CFG, None, None, bounds)
    casted = sel(bt.col("x").cast("float64") < 50.0, ndv, _CFG, None, None, bounds)
    assert casted == pytest.approx(plain)
    listed = sel(bt.col("x").cast("float64").is_in([1.0, 2.0]), ndv, _CFG, None, None, bounds)
    assert listed == pytest.approx(sel(bt.col("x").is_in([1, 2]), ndv, _CFG, None, None, bounds))


def test_an_opaque_or_lossy_cast_is_not_seen_through():
    """A boolean target collapses the domain, and `try_cast` changes the null fraction."""
    ndv = {"x": 100.0}
    assert sel(bt.col("x").cast("bool") == True, ndv, _CFG) == pytest.approx(  # noqa: E712
        _CFG.eq_selectivity
    )
    assert sel(bt.col("x").try_cast("float64") == 5.0, ndv, _CFG) == pytest.approx(
        _CFG.eq_selectivity
    )


def test_the_cast_estimate_tracks_the_executed_row_count():
    rows = 4000
    values = [i % 100 for i in range(rows)]
    learned = {"__column_ndv__": {"x": 100.0}}
    dataset = bt.from_pydict({"x": values}).filter(bt.col("x").cast("float64") == 5.0)
    stats = [s.statistics() for s in dataset._sources]
    estimated = (
        StatsEstimator(dataset._sources, learned, _CFG, source_stats=stats)
        .estimate(dataset._plan)
        .rows
    )
    executed = dataset.count()
    assert executed == rows // 100
    assert estimated == pytest.approx(executed, rel=0.05)
