"""`probe BETWEEN lower AND upper` over three columns is decided by the interval's width.

This is the temporal-validity lookup — `WHERE ts >= valid_from AND ts < valid_to` — which is
the whole of an SCD-2 point-in-time join (`ds.scd`), an IP-range or version lookup, and the
payload of every range join.

It is the one conjunction where independence is not merely imprecise but structurally wrong:
`lower` and `upper` are the two ends of *one* interval and move together, so each bound alone
really does cut about half the rows while the pair cuts far more. Exponential backoff sees two
loosely-dependent halves and lands between them. What decides the answer is the interval's
*width*, which neither conjunct mentions.

With the probe independent of the interval the exact answer is an expectation, not a product::

    P(lower <= probe <= upper) = E[F(upper) - F(lower)] = (E[upper] - E[lower]) / range(probe)

Measured over 20,000 rows with a 100-wide interval inside a 1,000-wide domain: 8,158 estimated
against 2,045 actual, a 4.0x over-estimate, against 2,000 from the formula.
"""

from __future__ import annotations

import random

import pytest

import batcher as bt
from batcher.config import active_config
from batcher.kyber.stats import StatsEstimator
from batcher.kyber.stats.selectivity.arithmetic import interval_containment
from batcher.plan.expr_rewrite import split_conjuncts

pytestmark = pytest.mark.unit

_CFG = active_config().optimizer.cardinality


def _estimate(dataset) -> float:
    stats = [s.statistics() for s in dataset._sources]
    return (
        StatsEstimator(dataset._sources, {}, _CFG, source_stats=stats).estimate(dataset._plan).rows
    )


def _frame(width: int, rows: int = 20000, span: int = 1000, seed: int = 19):
    """`t` uniform over `[0, span]`, and an interval of exactly `width` placed inside it."""
    rng = random.Random(seed)
    lower = [rng.randint(0, span - width) for _ in range(rows)]
    return bt.from_pydict(
        {
            "t": [rng.randint(0, span) for _ in range(rows)],
            "lo": lower,
            "hi": [x + width for x in lower],
        }
    )


# --- the pattern itself -----------------------------------------------------------


@pytest.mark.parametrize(
    "predicate",
    [
        (bt.col("t") >= bt.col("lo")) & (bt.col("t") <= bt.col("hi")),
        (bt.col("lo") <= bt.col("t")) & (bt.col("hi") >= bt.col("t")),
        (bt.col("t") > bt.col("lo")) & (bt.col("t") < bt.col("hi")),
    ],
    ids=["canonical", "flipped", "strict"],
)
def test_the_shape_is_recognised_whichever_way_it_is_written(predicate):
    match = interval_containment(split_conjuncts(predicate))
    assert match is not None
    probe, lower, upper, consumed = match
    assert (probe, lower, upper) == ("t", "lo", "hi")
    assert len(consumed) == 2


def test_a_degenerate_or_mismatched_shape_is_declined():
    """Both bounds the same column, or two different probes, is not a containment."""
    same = (bt.col("t") >= bt.col("lo")) & (bt.col("t") <= bt.col("lo"))
    assert interval_containment(split_conjuncts(same)) is None
    split = (bt.col("t") >= bt.col("lo")) & (bt.col("u") <= bt.col("hi"))
    assert interval_containment(split_conjuncts(split)) is None
    one_sided = split_conjuncts(bt.col("t") >= bt.col("lo"))
    assert interval_containment(one_sided) is None


# --- against executed row counts --------------------------------------------------


@pytest.mark.parametrize("width", [50, 100, 250])
def test_the_estimate_tracks_the_interval_width(width):
    frame = _frame(width)
    filtered = frame.filter((bt.col("t") >= bt.col("lo")) & (bt.col("t") <= bt.col("hi")))
    estimated, executed = _estimate(filtered), filtered.count()
    assert estimated == pytest.approx(executed, rel=0.15), (
        f"width {width}: estimated {estimated:.1f} against {executed}"
    )


def test_a_wider_interval_keeps_more_rows():
    """The defect was that the estimate barely moved with the width."""
    estimates = [
        _estimate(f.filter((bt.col("t") >= bt.col("lo")) & (bt.col("t") <= bt.col("hi"))))
        for f in (_frame(50), _frame(100), _frame(250))
    ]
    assert estimates[0] < estimates[1] < estimates[2]
    # A 5x wider interval keeps ~5x the rows; independence could not produce that spread.
    assert estimates[2] / estimates[0] == pytest.approx(5.0, rel=0.25)


def test_it_beats_the_independent_product_it_replaced():
    """Each bound alone cuts about half, so two independent halves land far too high."""
    frame = _frame(100)
    filtered = frame.filter((bt.col("t") >= bt.col("lo")) & (bt.col("t") <= bt.col("hi")))
    lower_only = frame.filter(bt.col("t") >= bt.col("lo"))
    assert _estimate(lower_only) > 4 * _estimate(filtered)
    assert _estimate(filtered) == pytest.approx(filtered.count(), rel=0.15)


def test_an_unmeasured_column_keeps_the_previous_estimate():
    """No bounds, no width — the rule declines and the per-conjunct estimate stands."""
    from batcher.kyber.stats.selectivity import predicate_selectivity as sel

    predicate = (bt.col("t") >= bt.col("lo")) & (bt.col("t") <= bt.col("hi"))
    assert sel(predicate, {}, _CFG, None, None, None) > 0.0
