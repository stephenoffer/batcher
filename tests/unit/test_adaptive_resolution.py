"""`adaptive="auto"`: the two gates that decide whether stage-by-stage re-opt runs.

`resolve_adaptive("auto", ...)` turns re-optimization on only when it could pay for itself,
and it asks two independent questions:

1. **Is the query big enough?** Re-opt trades a per-stage materialize and re-plan for a
   better downstream join choice. Below `_ADAPTIVE_MIN_INPUT_ROWS` the one-shot plan is
   already fast and the re-plan is pure overhead.
2. **Would measuring actually change anything?** Only a join whose operand comes out of a
   pipeline breaker with a *guessed* size (`Provenance.DEFAULT`) can have its build-side or
   join-order choice flipped by a real measurement. A join sized from source statistics
   gains nothing.

The size gate runs first and short-circuits. That matters for this file: with realistic
fixture tables (a handful of rows) *every* plan is below the threshold, so a test that just
asserts `False` passes without the confidence gate ever running — it would keep passing if
the confidence gate were deleted. So the confidence tests lower the threshold explicitly,
and one test pins the size gate on its own.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher.api.adaptive import gating as adaptive_mod
from batcher.api.adaptive import resolve_adaptive

pytestmark = pytest.mark.unit


def _hub():
    from batcher import core

    return core.default_hub()


def _fresh_hub():
    """An empty hub, so a test's own recorded history is the only thing the gate reads."""
    from batcher.metadata import MetadataHub
    from batcher.metadata.backends import InProcessBackend

    return MetadataHub(InProcessBackend())


@pytest.fixture
def any_size(monkeypatch):
    """Lower the size gate so the *confidence* gate is what the test measures."""
    monkeypatch.setattr(adaptive_mod, "_ADAPTIVE_MIN_INPUT_ROWS", 1)


def _join_over_a_breaker():
    """A join whose left operand is an aggregate output: a breaker, size only guessed."""
    left = bt.from_arrow(pa.table({"k": [1, 2, 3, 1, 2], "v": [10, 20, 30, 40, 50]}))
    right = bt.from_arrow(pa.table({"k": [1, 2, 3], "w": [100, 200, 300]}))
    return left.group_by("k").agg(s=col("v").sum()).join(right, on="k")


def test_auto_enables_for_join_over_uncertain_breaker(any_size):
    # Measured cardinality here flips a build-side / join-order choice, so re-opt earns
    # its cost — provided the query is big enough, which `any_size` stands in for.
    joined = _join_over_a_breaker()
    assert resolve_adaptive("auto", joined._plan, joined._sources, _hub()) is True


def test_auto_stays_one_shot_below_the_size_threshold():
    # The same plan, without lowering the gate. A few rows is not worth a re-plan, however
    # uncertain the operand is. This is the gate that makes the test above need `any_size`.
    joined = _join_over_a_breaker()
    assert resolve_adaptive("auto", joined._plan, joined._sources, _hub()) is False


def test_auto_stays_one_shot_without_join(any_size):
    # No join → re-optimization has no downstream decision to change.
    ds = bt.from_arrow(pa.table({"x": list(range(100))})).filter(col("x") > 5)
    assert resolve_adaptive("auto", ds._plan, ds._sources, _hub()) is False


def test_auto_stays_one_shot_for_scan_join_scan(any_size):
    # A join over two scans is sized from source statistics, not a guess — no benefit.
    sj = bt.from_arrow(pa.table({"k": [1, 2, 3], "a": [1, 2, 3]})).join(
        bt.from_arrow(pa.table({"k": [1, 2], "b": [9, 8]})), on="k"
    )
    assert resolve_adaptive("auto", sj._plan, sj._sources, _hub()) is False


def test_explicit_flag_always_wins():
    # Neither gate is consulted: an explicit choice is the caller's to make.
    joined = _join_over_a_breaker()
    assert resolve_adaptive(True, joined._plan, joined._sources, _hub()) is True
    assert resolve_adaptive(False, joined._plan, joined._sources, _hub()) is False


def test_auto_result_matches_one_shot():
    # Gating adaptivity trades planning overhead, never the result.
    left = bt.from_arrow(pa.table({"k": [1, 2, 3, 1, 2, 3], "v": [1, 2, 3, 4, 5, 6]}))
    right = bt.from_arrow(pa.table({"k": [1, 2, 3], "w": [10, 20, 30]}))

    def q():
        return left.group_by("k").agg(s=col("v").sum()).join(right, on="k")

    def norm(d):
        return sorted(zip(*[d[c] for c in sorted(d)], strict=True))

    auto = q().collect(adaptive="auto").to_pydict()
    one_shot = q().collect(adaptive=False).to_pydict()
    assert norm(auto) == norm(one_shot)


# --- The measured-history override ------------------------------------------------
#
# `Provenance.DEFAULT` says a size came from a Selinger guess. It does not say the guess
# was wrong, and on the one-shot path nothing ever records an intermediate operator's
# measured cardinality against its signature, so the label never clears. A shape can be
# estimated within a percent of actual forever and still fire the gate. These tests pin
# the correction: the label opens the question, the measured q-error history closes it.


def _feedback(signature: str, *, estimated: float, actual: int):
    from batcher.plan.feedback import OperatorFeedback

    return OperatorFeedback(
        op_id=1,
        kind="aggregate",
        n_actual=actual,
        t_op_ms=1.0,
        m_peak_bytes=0,
        selectivity=1.0,
        batch_size=1024,
        signature=signature,
        n_estimated=estimated,
    )


def _teach(hub, plan, *, ratio: float, runs: int) -> None:
    """Record `runs` executions of `plan`'s shape whose actual/estimated equals `ratio`."""
    from batcher.kyber.signature import plan_signature

    sig = plan_signature(plan)
    for _ in range(runs):
        hub.record(_feedback(sig, estimated=1000.0, actual=int(1000 * ratio)))


def test_accurate_history_turns_the_gate_off(any_size):
    # Same plan as the enabling test above, but its breaker operand now has a run of
    # executions whose estimates held. A stage boundary there would correct nothing, so
    # paying for one is pure cost.
    hub = _fresh_hub()
    joined = _join_over_a_breaker()
    assert resolve_adaptive("auto", joined._plan, joined._sources, hub) is True

    _teach(hub, joined._plan.left, ratio=1.02, runs=4)
    assert resolve_adaptive("auto", joined._plan, joined._sources, hub) is False


def test_inaccurate_history_leaves_the_gate_on(any_size):
    # A history that *did* cross the re-optimization threshold is exactly the case staging
    # exists for, so measured evidence must not suppress it.
    hub = _fresh_hub()
    joined = _join_over_a_breaker()
    _teach(hub, joined._plan.left, ratio=8.0, runs=4)
    assert resolve_adaptive("auto", joined._plan, joined._sources, hub) is True


def test_a_cold_hub_is_unchanged(any_size):
    # No history is not evidence of accuracy. A fresh hub must behave exactly as the gate
    # did before any of this existed, which is what keeps a first run's plan unchanged.
    joined = _join_over_a_breaker()
    assert resolve_adaptive("auto", joined._plan, joined._sources, _fresh_hub()) is True


def test_one_good_run_is_not_enough(any_size):
    # A single sample cannot distinguish an accurate estimator from a lucky one, so the
    # override waits for `cardinality_correction_min_samples`.
    hub = _fresh_hub()
    joined = _join_over_a_breaker()
    _teach(hub, joined._plan.left, ratio=1.0, runs=1)
    assert resolve_adaptive("auto", joined._plan, joined._sources, hub) is True


def test_alternating_errors_do_not_read_as_accurate(any_size):
    # The mean of a 4x over- and a 4x under-estimate is 1.0. Using it would call this
    # signature flawless, when it is precisely the shape a stage boundary corrects. The
    # check is on the worst sample, not the average.
    from batcher.kyber.signature import plan_signature

    hub = _fresh_hub()
    joined = _join_over_a_breaker()
    sig = plan_signature(joined._plan.left)
    for ratio in (4.0, 0.25, 4.0, 0.25):
        hub.record(_feedback(sig, estimated=1000.0, actual=int(1000 * ratio)))
    assert resolve_adaptive("auto", joined._plan, joined._sources, hub) is True


def test_the_override_does_not_change_the_answer(any_size):
    # The whole gate is a performance decision. Both routes return the identical relation,
    # so a hub that suppresses staging must not move a single row.
    left = bt.from_arrow(pa.table({"k": [1, 2, 3, 1, 2, 3], "v": [1, 2, 3, 4, 5, 6]}))
    right = bt.from_arrow(pa.table({"k": [1, 2, 3], "w": [10, 20, 30]}))

    def q():
        return left.group_by("k").agg(s=col("v").sum()).join(right, on="k")

    def norm(d):
        return sorted(zip(*[d[c] for c in sorted(d)], strict=True))

    _teach(_hub(), q()._plan.left, ratio=1.0, runs=4)
    assert norm(q().collect(adaptive="auto").to_pydict()) == norm(
        q().collect(adaptive=False).to_pydict()
    )
