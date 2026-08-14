"""The non-null mass is the budget a value distribution is spread over.

`ndv` counts distinct **non-null** values (`bc_sketches::HyperLogLog::add_array` skips
nulls), a quantile grid is built over the non-null values, and `[min, max]` bound them. So
every statistic-derived selectivity answers `P(predicate | col IS NOT NULL)`, and reporting
it unconditionally over-states a predicate over a nullable column by exactly
`1 / (1 - f_null)` — 1.4x at 30% null, 10x at 90%.

That error was one-sided and therefore invisible to the identity the complement side already
enforced: `!=` and `NOT` subtracted the null mass (`_null_mass`) while `=`, `IN`, `<`, `LIKE`
and the bare boolean column did not, so a predicate and its negation disagreed about how many
rows the relation even had. The estimates feed join order, build-side choice and broadcast
sizing, which is why this is measured here against *executed* row counts rather than against
a constant.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.config import active_config
from batcher.kyber.stats import StatsEstimator
from batcher.kyber.stats.selectivity import predicate_selectivity as sel

pytestmark = pytest.mark.unit

_CFG = active_config().optimizer.cardinality
_NDV = {"x": 10.0}
_QUANTILES = {"x": {"probs": [0.0, 1.0], "values": [0.0, 10.0]}}
_NULLS = {"x": 0.3}
_NON_NULL = 0.7


def _s(expr, *, nulls=None, ndv=None, mcv=None):
    return sel(expr, ndv if ndv is not None else _NDV, _CFG, _QUANTILES, mcv, None, nulls)


# --- the identity: a predicate and its negation partition the non-null rows -------


@pytest.mark.parametrize(
    "predicate",
    [
        bt.col("x") == 5,
        bt.col("x") < 5,
        bt.col("x") >= 5,
        bt.col("x").is_in([1, 2, 3]),
        bt.col("x").str.contains("a"),
    ],
    ids=["eq", "lt", "ge", "in", "contains"],
)
def test_a_predicate_and_its_negation_partition_the_non_null_rows(predicate):
    positive = _s(predicate, nulls=_NULLS)
    negated = _s(~predicate, nulls=_NULLS)
    assert positive + negated == pytest.approx(_NON_NULL)
    # Neither half may claim rows the column has no value on.
    assert 0.0 <= positive <= _NON_NULL
    assert 0.0 <= negated <= _NON_NULL


@pytest.mark.parametrize(
    "predicate",
    [bt.col("x") == 5, bt.col("x") < 5, bt.col("x").is_in([1, 2, 3])],
    ids=["eq", "lt", "in"],
)
def test_the_null_budget_scales_the_conditional_estimate(predicate):
    """The nullable estimate is exactly the null-free one times the non-null fraction."""
    assert _s(predicate, nulls=_NULLS) == pytest.approx(_NON_NULL * _s(predicate))


def test_an_unmeasured_null_count_changes_nothing():
    """No measurement means no budget: an unmeasured column is treated as null-free."""
    for predicate in (bt.col("x") == 5, bt.col("x") < 5, bt.col("x").is_in([1, 2])):
        assert _s(predicate, nulls={}) == pytest.approx(_s(predicate))
        assert _s(predicate, nulls={"other": 0.9}) == pytest.approx(_s(predicate))


def test_a_measured_mcv_frequency_is_not_scaled_twice():
    """An MCV frequency is measured as a share of *all* rows, so it is already unconditional.

    Scaling it by the budget as well would subtract the null rows a second time.
    """
    mcv = {"x": {"5": 0.42}}
    assert _s(bt.col("x") == 5, nulls=_NULLS, mcv=mcv) == pytest.approx(0.42)
    assert _s(bt.col("x") == 5, mcv=mcv) == pytest.approx(0.42)


def test_is_null_is_never_scaled():
    """`IS NULL` is two-valued — it is *about* the null rows, not dropped by them."""
    assert _s(bt.col("x").is_null(), nulls=_NULLS) == pytest.approx(0.3)
    assert _s(bt.col("x").is_not_null(), nulls=_NULLS) == pytest.approx(_NON_NULL)


def test_an_all_null_column_keeps_nothing():
    """With no non-null rows there is no budget, so no positive predicate can match."""
    for predicate in (bt.col("x") == 5, bt.col("x") < 5, bt.col("x").is_in([1, 2])):
        assert _s(predicate, nulls={"x": 1.0}) == pytest.approx(0.0)


# --- against executed row counts -------------------------------------------------


def _estimate(ds) -> float:
    stats = [s.statistics() for s in ds._sources]
    return StatsEstimator(ds._sources, {}, _CFG, source_stats=stats).estimate(ds._plan).rows


@pytest.mark.parametrize(
    ("name", "build"),
    [
        ("eq", lambda: bt.col("v") == 7),
        ("lt", lambda: bt.col("v") < 25),
        ("ge", lambda: bt.col("v") >= 25),
        ("between", lambda: (bt.col("v") >= 10) & (bt.col("v") <= 20)),
        ("in", lambda: bt.col("v").is_in([1, 2, 3, 4, 5])),
    ],
)
def test_the_estimate_tracks_the_executed_row_count_over_a_nullable_column(name, build):
    """A third of the rows are NULL; the estimate must not hand those to the predicate.

    The data is deterministic and uniform, so the surviving count is what uniformity
    predicts and the tolerance can be tight. Before the budget existed every case here
    over-estimated by `1 / (1 - 1/3)` = 1.5x.
    """
    rows = 3000
    values = [None if i % 3 == 0 else i % 50 for i in range(rows)]
    dataset = bt.from_pydict({"v": values}).filter(build())
    estimated, executed = _estimate(dataset), dataset.count()
    assert executed > 0
    assert estimated == pytest.approx(executed, rel=0.25), (
        f"{name}: estimated {estimated:.1f} against {executed} executed rows"
    )
    # The specific failure this guards: the old estimate was the actual times 1/(1 - f_null).
    assert estimated < executed * 1.4
