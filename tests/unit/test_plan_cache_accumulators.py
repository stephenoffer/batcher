"""The plan cache's accumulator lists must describe the state its writers actually store.

`plan_cache._materially_differs` decides whether a learned write invalidates every memoized
plan. It classifies a dict's fields by *name*: a field named in `_BOOKKEEPING_FIELDS` is an
accumulator that grows with every observation and is ignored, a pair in `_DERIVED_RATIOS` is
compared as a quotient, and anything else is compared directly.

Names are the whole weakness. When `kyber.ols` was rewritten from power sums
(`sx`/`sy`/`sxx`/`sxy`) to a centered Welford form (`mx`/`my`/`m2x`/`m2y`/`cxy`), and
`bandit.record_arm` from `sum`/`sumsq` to a discounted Welford `(n, mean, m2)`, both lists kept
naming the retired fields. Nothing failed and no test noticed: the writes simply started
looking material on every execution, and the plan cache was flushed by the learning loop on
essentially every join. Measured on TPC-H at scale 1, seven of twenty-two queries never hit the
memo at all, and those were exactly the queries whose control plane dominated their wall clock.

So these tests hold the lists against the writers rather than against a fixture: they run the
real `ols_update` and the real `record_arm`, and assert that every field those produce is one
the classifier has an opinion about. A future rename fails here instead of quietly costing a
third of the wall clock.
"""

from __future__ import annotations

import pytest

from batcher.kyber import ols
from batcher.kyber.plan_cache import (
    _BOOKKEEPING_FIELDS,
    _DERIVED_RATIOS,
    _materially_differs,
)

pytestmark = pytest.mark.unit

#: Fields a plan genuinely reads as a value, so comparing them directly is correct.
_DECISION_FIELDS = frozenset({"mean", "mx", "my", "xmin", "xmax"})


def _classified(field: str) -> bool:
    """Whether `field` is one the materiality test has been taught about."""
    ratio_fields = {f for pair in _DERIVED_RATIOS for f in pair}
    return field in _BOOKKEEPING_FIELDS or field in ratio_fields or field in _DECISION_FIELDS


def test_every_ols_field_is_classified():
    state = {}
    for x, y in ((10.0, 1.0), (20.0, 2.5), (30.0, 3.1)):
        state = ols.ols_update(state, x, y)
    unclassified = sorted(f for f in state if not _classified(f))
    assert not unclassified, (
        f"`kyber.ols` writes {unclassified}, which `plan_cache` has never heard of. "
        "Classify each in `_BOOKKEEPING_FIELDS`, `_DERIVED_RATIOS`, or as a decision field — "
        "an unclassified accumulator makes every write look material and flushes the memo."
    )


def test_an_ols_observation_that_moves_nothing_material_does_not_invalidate():
    """Two identical observations move `n` and the co-moments, and no decision."""
    first = ols.ols_update(ols.ols_update({}, 10.0, 1.0), 20.0, 2.0)
    second = ols.ols_update(first, 15.0, 1.5)  # exactly on the fitted line, at the mean
    assert not _materially_differs(first, second)


def test_every_bandit_arm_field_is_classified():
    from batcher.kyber.learned_tuning.bandit import _welford_update

    state = _welford_update(_welford_update(None, 10.0), 12.0)
    unclassified = sorted(f for f in state if not _classified(f))
    assert not unclassified, (
        f"`record_arm` writes {unclassified}, which `plan_cache` has never heard of; see "
        "`test_every_ols_field_is_classified` for why that is expensive rather than harmless."
    )


def test_a_bandit_write_that_only_decays_does_not_invalidate():
    """Decay alone scales `n` and `m2` together, leaving the mean and the variance put."""
    from batcher.kyber.learned_tuning.bandit import _decayed, _welford_update

    arm = _welford_update(_welford_update(None, 10.0), 10.0)
    assert not _materially_differs({"a": arm}, {"a": _decayed(arm)})
