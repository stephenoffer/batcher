"""The Kyber → Carbonite → Core contract loop, at the seams where it was broken.

Each test pins one invariant the loop *claims* and did not hold:

* **Kyber decides, Carbonite protects** — but Carbonite must not reject a plan on a
  `Provenance.DEFAULT` guess. `plan/physical.py` promises provenance is read here; it was
  not, so an unmeasurable plan could be failed outright.
* **Core measures, Kyber re-decides** — the adaptive loop's accuracy test must be
  symmetric. A relative error normalized by the estimate is bounded by 1 for *any*
  over-estimate, so it declared every over-estimate accurate and stopped re-optimizing —
  disabling the loop for exactly the case it exists to catch.
* **Carbonite's flow control must be a control law** — AIMD's multiplicative decrease is
  only a decrease for `0 < beta < 1`.
* **Recovery must not pay for work it discards.**
"""

from __future__ import annotations

import pytest

from batcher.api.adaptive import _estimate_accurate
from batcher.carbonite.base import ResourceContext
from batcher.carbonite.policies import AIMDFlowControl, BudgetingAdmission
from batcher.config import active_config
from batcher.plan.ids import OpId
from batcher.plan.physical import PhysicalOp, PhysicalPlan, PlanProperties
from batcher.plan.resource import ResourceBounds
from batcher.plan.stats import Provenance

pytestmark = pytest.mark.unit


# --- adaptive re-optimization: the accuracy gate ----------------------------------

_REOPT_ERROR = 2.0  # the shipped default: accurate within a factor of 3


def test_large_over_estimate_is_not_accurate():
    """The regression this loop exists for.

    A selective filter feeding a join produces far fewer rows than estimated; that is what
    makes the join's operand small enough to broadcast. The old gate computed
    `|actual - est| / est`, which for `actual <= est` can never exceed 1 — so with
    `reoptimize_error = 2.0` every over-estimate compared "accurate" and the loop stopped.
    """
    assert not _estimate_accurate(actual=1, estimate=1_000_000, reopt_error=_REOPT_ERROR)


def test_accuracy_band_is_symmetric_in_q_error():
    # Accurate exactly when actual in [est/3, 3*est]: cardinality error is multiplicative.
    assert _estimate_accurate(1000, 3000, _REOPT_ERROR)  # 3x over
    assert _estimate_accurate(3000, 1000, _REOPT_ERROR)  # 3x under
    assert not _estimate_accurate(1000, 3001, _REOPT_ERROR)
    assert not _estimate_accurate(3001, 1000, _REOPT_ERROR)


def test_exact_estimate_is_accurate():
    assert _estimate_accurate(1000, 1000, _REOPT_ERROR)


def test_degenerate_estimates_are_never_accurate():
    assert not _estimate_accurate(0, 1000, _REOPT_ERROR)  # produced nothing; a total miss
    assert not _estimate_accurate(1000, 0, _REOPT_ERROR)  # no estimate to compare against
    assert not _estimate_accurate(-1, 1000, _REOPT_ERROR)


# --- admission: never reject on a guess -------------------------------------------


def _one_op_plan(provenance: Provenance, peak_bytes: int) -> PhysicalPlan:
    op = PhysicalOp(
        op_id=OpId(0),
        kind="Aggregate",
        backend="native",
        algorithm="",
        bounds=ResourceBounds(peak_bytes, 0, 0),
        inputs=(),
        properties=PlanProperties(est_rows=1e9, provenance=provenance),
    )
    return PhysicalPlan(ir={}, output_schema=None, ops=(op,))


def _validate(provenance: Provenance):
    ctx = ResourceContext(config=active_config(), envelope_bytes=1_000_000)
    admission = BudgetingAdmission(available_bytes=1_000_000, soft_limit=0.85)
    return admission.validate(_one_op_plan(provenance, 10**9), ctx)


def test_infeasibility_from_a_guess_is_advisory():
    # It still routes the plan out-of-core; it must not be allowed to fail the query.
    verdict = _validate(Provenance.DEFAULT)
    assert verdict.feasible is False
    assert verdict.advisory is True


@pytest.mark.parametrize(
    "provenance", [Provenance.EXACT, Provenance.HISTOGRAM, Provenance.SKETCH, Provenance.LEARNED]
)
def test_infeasibility_from_evidence_is_binding(provenance):
    # A proof, a footer, a sketch, or a past measurement are all trusted to reject.
    verdict = _validate(provenance)
    assert verdict.feasible is False
    assert verdict.advisory is False


def test_a_plan_that_fits_is_feasible_whatever_its_provenance():
    ctx = ResourceContext(config=active_config(), envelope_bytes=1_000_000)
    admission = BudgetingAdmission(available_bytes=1_000_000, soft_limit=0.85)
    verdict = admission.validate(_one_op_plan(Provenance.DEFAULT, 1024), ctx)
    assert verdict.feasible is True


# --- flow control: AIMD must actually be a control law -----------------------------


def _aimd_with_beta(beta: float) -> AIMDFlowControl:
    import dataclasses

    cfg = active_config()
    cfg = dataclasses.replace(
        cfg, flow_control=dataclasses.replace(cfg.flow_control, aimd_beta=beta)
    )
    # Start well below the ceiling so an additive increase is observable.
    return AIMDFlowControl(cfg, initial_window=4)


@pytest.mark.parametrize("beta", [1.0, 1.5, 42.0])
def test_congestion_always_decreases_the_window_even_with_a_bad_beta(beta):
    # `beta >= 1` would turn multiplicative *decrease* into growth — an unstable law that
    # grows the window on congestion. The controller clamps rather than failing a query.
    aimd = _aimd_with_beta(beta)
    before = aimd.window
    after = aimd.observe(congested=True)
    assert after < before


@pytest.mark.parametrize("beta", [0.0, -1.0])
def test_a_non_positive_beta_cannot_collapse_the_window(beta):
    aimd = _aimd_with_beta(beta)
    after = aimd.observe(congested=True)
    assert after >= 1


def test_uncongested_rounds_grow_the_window():
    aimd = _aimd_with_beta(0.5)
    assert aimd.observe(congested=False) > 4


# --- resilience: no recompute whose result is discarded ---------------------------


def test_recovery_does_not_recompute_on_the_final_round():
    from batcher._internal.errors import ResourceError
    from batcher.carbonite.resilience.recovery import ShuffleRecovery

    class _Policy:
        max_attempts = 3
        backoff_base_s = 0.0

    recovery = ShuffleRecovery(_Policy())
    recomputes: list[object] = []

    def attempt():
        return None, ["p0"]  # always fails

    with pytest.raises(ResourceError):
        recovery.run(attempt, recomputes.append)
    # 3 attempts, but only the 2 that are followed by another attempt may recompute.
    assert len(recomputes) == 2
    assert recovery.recomputes == 2


# --- scheduling: a memory reservation must be what the task needs -------------------


def test_per_task_memory_hint_is_not_divided_by_the_cluster_fan_out():
    """Ray's `memory=` is a reservation; under-reporting it over-packs the node.

    The plan's peak is already divided by the task count (each task holds one partition).
    The old clamp divided a *single machine's* budget by the *cluster-wide* fan-out and
    took the min, so a 100-task job reserved 1/100th of a node per task and Ray stacked
    all hundred onto one node.
    """
    from batcher.carbonite.policies import DefaultSchedulingPolicy

    gib = 2**30
    peak, tasks, driver_budget = 100 * gib, 100, 8 * gib
    op = PhysicalOp(
        op_id=OpId(0),
        kind="Aggregate",
        backend="native",
        algorithm="",
        bounds=ResourceBounds(peak, 0, tasks),
        inputs=(),
        properties=PlanProperties(est_rows=1e9, provenance=Provenance.LEARNED),
    )
    plan = PhysicalPlan(ir={}, output_schema=None, ops=(op,))
    ctx = ResourceContext(config=active_config(), envelope_bytes=driver_budget)

    envelope = DefaultSchedulingPolicy().envelope(
        plan, ctx, requested_workers=None, available_bytes=driver_budget
    )
    assert envelope.memory_bytes == peak // tasks  # exactly what one partition needs
    assert envelope.memory_bytes > driver_budget // tasks  # not the old fair share


def test_per_task_memory_hint_never_exceeds_one_node():
    # Asking for more than a node holds makes the task permanently unschedulable.
    from batcher.carbonite.policies import DefaultSchedulingPolicy

    gib = 2**30
    op = PhysicalOp(
        op_id=OpId(0),
        kind="Aggregate",
        backend="native",
        algorithm="",
        bounds=ResourceBounds(100 * gib, 0, 1),  # one task holding the whole 100 GiB
        inputs=(),
        properties=PlanProperties(est_rows=1e9, provenance=Provenance.LEARNED),
    )
    plan = PhysicalPlan(ir={}, output_schema=None, ops=(op,))
    ctx = ResourceContext(config=active_config(), envelope_bytes=8 * gib)
    envelope = DefaultSchedulingPolicy().envelope(
        plan, ctx, requested_workers=None, available_bytes=8 * gib
    )
    assert envelope.memory_bytes <= 8 * gib


# --- the broadcast-vs-shuffle crossover actually learns -----------------------------


def test_broadcast_crossover_learns_from_measured_timings():
    """`record_broadcast_timing` had no callers, so `learned_broadcast_max_bytes` was
    permanently `None` and the threshold never moved off its static default."""
    from batcher.kyber.learned_tuning import learned_broadcast_max_bytes, record_broadcast_timing
    from batcher.metadata import MetadataHub
    from batcher.metadata.backends.in_process import InProcessBackend

    hub = MetadataHub(InProcessBackend())
    assert learned_broadcast_max_bytes(hub) is None  # cold

    mib = 2**20
    # broadcast: cheap fixed cost, replication grows fast. shuffle: costly fixed, grows slow.
    # They cross at 5 + 1.2x = 50 + 0.3x  ->  x = 50 MiB.
    for mb in (1, 2, 4, 8, 16, 32, 48, 64, 80, 96):
        record_broadcast_timing(hub, "broadcast", mb * mib, 5.0 + 1.2 * mb)
        record_broadcast_timing(hub, "shuffle", mb * mib, 50.0 + 0.3 * mb)

    learned = learned_broadcast_max_bytes(hub)
    assert learned is not None
    assert learned / mib == pytest.approx(50.0, abs=1.0)
