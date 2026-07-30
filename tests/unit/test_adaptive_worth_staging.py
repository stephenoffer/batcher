"""Which breakers the adaptive loop stages — the predicate, at its boundary.

Staging trades a materialization for a measurement. That trade returns nothing only when
the optimizer already knows the breaker's output size **exactly**; against any weaker
estimate the measurement is the entire point of the loop.

`Provenance` is an `IntEnum` ordered *strongest-trust first* — ``EXACT = 0`` through
``DEFAULT = 4`` — so "less than exact" is ``> EXACT``. Written as ``>= DEFAULT`` it means
something very different: only the pure Selinger guess qualifies, and `HISTOGRAM`,
`SKETCH` and `LEARNED` breakers all run inline unmeasured. That was the bug. Its effect
was perverse — the better Batcher's learned statistics got, the *less* it re-optimized
adaptively, because a `LEARNED` estimate stopped qualifying for measurement — and it is
invisible in any result comparison, since staging changes performance and decision quality
rather than answers.

`tests/differential/test_diff_adaptive.py::test_adaptive_uses_exact_cardinalities` catches
it end to end (it went from `stages=1, provenance=learned` to `stages=2, provenance=exact`
when this was fixed). This file pins the predicate itself, at the one boundary that
matters, so a future edit cannot re-introduce the same off-by-one-enum.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.api.adaptive.staging import _worth_staging
from batcher.plan.stats import Provenance

pytestmark = pytest.mark.unit


class _Estimate:
    def __init__(self, provenance: Provenance):
        self.provenance = provenance


class _Estimator:
    """Answers every node with one fixed provenance."""

    def __init__(self, provenance: Provenance):
        self._provenance = provenance

    def estimate(self, node):
        return _Estimate(self._provenance)


@pytest.fixture
def plan_node():
    return bt.from_arrow(pa.table({"v": [1, 2, 3]})).group_by("v").agg(n=bt.count())._plan


def _accepts(monkeypatch, provenance: Provenance, node) -> bool:
    """Whether the predicate would stage a breaker estimated with `provenance`."""
    import batcher.api.adaptive.gating as gating

    monkeypatch.setattr(gating, "_build_estimator", lambda srcs, hub: _Estimator(provenance))
    return _worth_staging([], None)(node)


def test_an_exactly_sized_breaker_is_not_staged(monkeypatch, plan_node):
    """The optimization this predicate exists for: measuring what is already known is waste."""
    assert not _accepts(monkeypatch, Provenance.EXACT, plan_node)


@pytest.mark.parametrize(
    "provenance",
    [Provenance.HISTOGRAM, Provenance.SKETCH, Provenance.LEARNED, Provenance.DEFAULT],
)
def test_every_weaker_estimate_is_staged(monkeypatch, provenance, plan_node):
    """Anything short of exact is worth measuring — including `LEARNED`.

    `LEARNED` is the one that matters most: it is what a warmed-up process produces for
    almost every breaker, so excluding it disabled the loop in exactly the steady state
    the learned-statistics work exists to reach.
    """
    assert _accepts(monkeypatch, provenance, plan_node), (
        f"a {provenance} breaker was not staged; the loop can only correct an estimate it "
        f"measures, and {provenance} is not exact"
    )


def test_the_boundary_is_exact_and_not_default(monkeypatch, plan_node):
    """Stated as the comparison, so the enum's direction cannot be misread again.

    `Provenance` sorts strongest-first, which makes `>= DEFAULT` a *narrower* filter than
    `> EXACT` rather than a wider one — the reading that produced the bug.
    """
    assert Provenance.EXACT < Provenance.LEARNED < Provenance.DEFAULT, (
        "Provenance must stay ordered strongest-trust first, or this predicate inverts"
    )
    staged = {
        p
        for p in Provenance
        if _accepts(monkeypatch, p, plan_node)
    }
    assert staged == set(Provenance) - {Provenance.EXACT}, (
        f"staged provenances {sorted(p.name for p in staged)}; expected everything except EXACT"
    )


def test_an_estimator_that_raises_still_stages(monkeypatch, plan_node):
    """Best-effort by design: a failed estimate must never cost the loop a measurement."""
    import batcher.api.adaptive.gating as gating

    def _boom(srcs, hub):
        raise RuntimeError("estimator unavailable")

    monkeypatch.setattr(gating, "_build_estimator", _boom)
    assert _worth_staging([], None)(plan_node), (
        "an unreadable estimate must fall back to staging, not to skipping"
    )
