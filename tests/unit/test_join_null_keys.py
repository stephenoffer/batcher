"""An equi-join matches only rows whose key holds a value.

`NULL = NULL` is not true, so a null key joins to nothing. The estimator counted them anyway:
the MCV decomposition spread each side's residual over `1 - Σ f` — the whole relation, null
rows included — and the Selinger ratio met two full row counts. The result did not move *at
all* as the null fraction rose. Measured on a single-key join over 8,000 x 2,000 rows:

    null fraction   estimated   actual
    0.0             171,817     160,867
    0.3             172,878      79,039
    0.6             172,134      26,181

so a join on two 60%-null keys was over-estimated 6.6x — exactly `1/((1-f_L)(1-f_R))`. A
nullable join key is the ordinary case for an optional foreign key, an outer-join result, or a
sparse dimension, and the estimate drives build-side choice, join order and memory admission.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.config import active_config
from batcher.kyber.stats import StatsEstimator
from batcher.kyber.stats.distribution import mcv_join_rows

pytestmark = pytest.mark.unit

_CFG = active_config().optimizer.cardinality


def _estimate(dataset) -> float:
    stats = [s.statistics() for s in dataset._sources]
    return (
        StatsEstimator(dataset._sources, {}, _CFG, source_stats=stats).estimate(dataset._plan).rows
    )


def _sides(null_every: int, rows: int = 2000, keys: int = 50):
    """Two relations whose key is NULL on every `null_every`-th row (0 disables nulls).

    `null_every` is chosen coprime-ish to `keys` so the surviving rows still cover the whole
    key domain — otherwise the fixture, not the estimator, decides the distinct count.
    """

    def build(n):
        return bt.from_pydict(
            {"k": [None if null_every and i % null_every == 0 else i % keys for i in range(n)]}
        )

    return build(rows), build(rows // 2)


@pytest.mark.parametrize("null_every", [0, 4, 3])
def test_the_estimate_tracks_the_null_fraction(null_every):
    left, right = _sides(null_every)
    joined = left.join(right, on="k")
    estimated, executed = _estimate(joined), joined.count()
    assert executed > 0
    assert estimated == pytest.approx(executed, rel=0.35), (
        f"1/{null_every} null: estimated {estimated:.1f} against {executed}"
    )


def test_more_nulls_means_fewer_estimated_rows():
    """The defect was that this was *flat*: the estimate ignored the null count entirely."""
    pairs = (_sides(0), _sides(4), _sides(3))
    estimates = [_estimate(left.join(right, on="k")) for left, right in pairs]
    assert estimates[0] > estimates[1] > estimates[2]


def test_a_nearly_all_null_key_estimates_close_to_nothing():
    """A key that is null on all but a handful of rows can match on no more than those.

    Deliberately not *entirely* null: an all-`None` column is Arrow type `null`, so the join
    inserts a genuine type-changing cast to align it and `project_columns` drops the statistic
    — correctly, since a value-changing cast's output distribution is not the input's.
    """
    left = bt.from_pydict({"k": [1 if i < 5 else None for i in range(500)]})
    right = bt.from_pydict({"k": list(range(50))})
    joined = left.join(right, on="k")
    assert joined.count() == 5
    assert _estimate(joined) < 50.0, "the null rows were still being matched"


def test_the_residual_uses_the_non_null_mass():
    """`mcv_join_rows` spreads each side's residual over its non-null mass, not over 1."""
    left_mcv = {"a": 0.1, "b": 0.1}
    right_mcv = {"a": 0.1, "b": 0.1}
    full = mcv_join_rows(1000.0, 1000.0, left_mcv, right_mcv, 20.0, 20.0)
    halved = mcv_join_rows(1000.0, 1000.0, left_mcv, right_mcv, 20.0, 20.0, 0.5, 0.5)
    assert halved < full
    # The default keeps the previous behaviour exactly, for a caller with no null count.
    assert mcv_join_rows(1000.0, 1000.0, left_mcv, right_mcv, 20.0, 20.0, 1.0, 1.0) == full


def test_a_composite_key_needs_every_column_non_null():
    """A row matches only when the whole key holds a value, so the smallest share binds."""
    from batcher.kyber.stats.estimator import _key_non_null
    from batcher.plan.stats import ColumnStat, Provenance, RelStats

    stats = RelStats(
        100.0,
        Provenance.EXACT,
        {
            "a": ColumnStat(null_count=10.0, provenance=Provenance.EXACT),
            "b": ColumnStat(null_count=40.0, provenance=Provenance.EXACT),
        },
    )
    assert _key_non_null(("a", "b"), stats) == pytest.approx(0.6)
    assert _key_non_null(("a",), stats) == pytest.approx(0.9)
    # An unmeasured column contributes nothing, as the rest of the estimator assumes.
    assert _key_non_null(("a", "missing"), stats) == pytest.approx(0.9)
    assert _key_non_null((), stats) == pytest.approx(1.0)
