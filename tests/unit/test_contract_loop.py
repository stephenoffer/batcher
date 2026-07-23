"""The Kyber → Carbonite → Core contract loop, at the seams where it was broken.

Each test pins one invariant the loop *claims* and did not hold:

* **Kyber decides, Carbonite protects** — but Carbonite must not reject a plan on a
  `Provenance.DEFAULT` guess. `plan/physical.py` promises provenance is read here; it was
  not, so an unmeasurable plan could be failed outright.
* **Core measures, Kyber re-decides** — the adaptive loop's accuracy test must be
  symmetric. A relative error normalized by the estimate is bounded by 1 for *any*
  over-estimate, so it declared every over-estimate accurate and stopped re-optimizing —
  disabling the loop for exactly the case it exists to catch.
* **Carbonite counter-offers, and the offer binds** — an infeasible verdict carries the
  envelope the plan *would* fit in. It was dropped, so the conductor sharded the spill by a
  fixed constant that knows nothing about the machine's budget.
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


def test_the_counter_offer_binds_the_spill_sharding():
    """An infeasible verdict's `suggested_bounds` must reach the out-of-core sharding.

    This is the documented return leg of the Kyber<->Carbonite contract: admission does not
    merely refuse a plan, it names the per-operator envelope the plan *would* fit in. Nothing
    consumed `suggested_bounds`, so the conductor degraded the verdict to a bare `must_spill`
    boolean and sharded by a fixed bytes-per-partition constant that knows nothing about the
    machine's budget — producing buckets that individually still do not fit, which is exactly
    what admission had just diagnosed. Partition count is result-invariant (mergeable
    algebra), so this is purely a memory-safety lever.
    """
    from batcher.carbonite import ResourceManager

    rm = ResourceManager()
    peak = 8 << 30
    op = PhysicalOp(
        op_id=OpId(1),
        kind="Aggregate",
        backend="native",
        algorithm="",
        bounds=ResourceBounds(m_max_bytes=peak, c_max_credits=0, n_max_parallelism=0),
        inputs=(),
        properties=PlanProperties(est_rows=1e8),
    )
    plan = PhysicalPlan(ir={}, output_schema=None, ops=(op,))

    # No counter-offer (a feasible plan) leaves the caller's own count untouched.
    assert rm.partitions_for_bounds(plan, None) == 0

    # An offer of E must shard an 8 GiB peak into ceil(8 GiB / E) buckets, so each fits.
    for envelope_gib, expected in ((1, 8), (2, 4), (4, 2)):
        offer = ResourceBounds(m_max_bytes=envelope_gib << 30, c_max_credits=0, n_max_parallelism=0)
        assert rm.partitions_for_bounds(plan, offer) == expected
    # A tighter envelope shards further, never coarser.
    tight = ResourceBounds(m_max_bytes=256 << 20, c_max_credits=0, n_max_parallelism=0)
    assert rm.partitions_for_bounds(plan, tight) == 32


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

    # The solved crossover is clamped to a band around the *default*, so pass the
    # default explicitly: this asserts the value is learned from the timings, not
    # that it happens to sit inside the band of whatever the config default is today.
    learned = learned_broadcast_max_bytes(hub, default=10 * mib)
    assert learned is not None
    assert learned / mib == pytest.approx(50.0, abs=1.0)


# --- the optimizer driver: a once-run phase must not lose a rewrite ----------------


def test_a_once_run_phase_visits_a_subtree_a_rule_just_created():
    """`_apply_rules` fused node rules into one bottom-up walk and claimed equivalence.

    It is not equivalent: `transform_up` has already visited a node's children by the time
    a rule fires on it, so a rule that rewrites a node into a *new subtree* leaves those
    new children unvisited by the later rules in the same run. A fixpoint phase recovers
    them next iteration. SELECTION and ENFORCE run once — the rewrite was lost outright.

    Here `wrap` turns `Limit(x, 1)` into `Limit(Limit(x, 1), 1)`, and `mark` rewrites any
    `Limit(_, 1)` into `Limit(_, 7)`. Running them fused in a once-run phase leaves the
    freshly-created inner limit at 1.
    """
    import batcher as bt
    from batcher.kyber.optimizer.driver import _apply_rules
    from batcher.kyber.pass_base import OptimizerContext
    from batcher.kyber.rule import Phase, node_rule
    from batcher.kyber.stats import StatsEstimator
    from batcher.plan.logical import Limit

    wrapped: list[int] = []

    def wrap(node, _ctx):
        if node.n == 1 and not wrapped:
            wrapped.append(1)
            return Limit(node, 1)
        return None

    def mark(node, _ctx):
        return Limit(node.input, 7) if node.n == 1 else None

    rules = [
        node_rule("wrap", Phase.SELECTION, wrap, matches=(Limit,)),
        node_rule("mark", Phase.SELECTION, mark, matches=(Limit,)),
    ]
    cfg = active_config()
    ctx = OptimizerContext(
        config=cfg,
        sources=[],
        hub=None,
        estimator=StatsEstimator([], {}, cfg.optimizer.cardinality),
    )
    plan = bt.from_pydict({"x": [1, 2, 3]}).limit(1)._plan

    fused = _apply_rules(plan, rules, ctx, fuse=True)
    wrapped.clear()
    unfused = _apply_rules(plan, rules, ctx, fuse=False)

    # Fused: `mark` never sees the inner limit `wrap` just created.
    assert fused.n == 7 and fused.input.n == 1
    # Unfused (what a once-run phase now does): every limit is marked.
    assert unfused.n == 7 and unfused.input.n == 7


def test_fixpoint_phases_still_fuse():
    # Fusing is sound where a next iteration recovers the missed rewrite, and it is the
    # hot path (NORMALIZE/REWRITE/PUSHDOWN/FUSION hold most of the rule set).
    import inspect

    from batcher.kyber.optimizer.driver import _run_phase

    source = inspect.getsource(_run_phase)
    assert "fuse = max_iterations > 1" in source


# --- pressure hysteresis must not depend on how often it is read -------------------


def test_reading_the_pressure_level_does_not_advance_its_hysteresis():
    """`level()` is the *sampler*: each call folds one reading into the de-escalation EWMA.

    Three components called it per round at unrelated cadences, so the average advanced
    several steps and collapsed toward the raw reading — defeating the anti-flap smoothing
    that stops SPILL<->NORMAL oscillation from thrashing the shuffle's AIMD credit window.
    Readers now use `classify()`, which is pure.
    """
    from batcher.carbonite.memory.pressure import PressureMonitor

    monitor = PressureMonitor(active_config())
    readings = iter([0.95] + [0.10] * 20)  # one spike, then sustained calm
    monitor._engine_used_fraction = staticmethod(lambda: next(readings, 0.10))

    monitor.level()  # sample the spike
    spiked = monitor._ewma

    for _ in range(10):
        monitor.classify()
    assert monitor._ewma == spiked, "a pure read must not advance the average"

    before = monitor._ewma
    monitor.level()  # one sample of the calm reading
    assert monitor._ewma < before, "the sampler still decays it"


def test_the_sampler_decays_one_step_per_call():
    from batcher.carbonite.memory.pressure import PressureMonitor

    monitor = PressureMonitor(active_config())
    monitor._engine_used_fraction = staticmethod(lambda: 0.0)
    monitor._ewma = 1.0

    monitor.level()
    after_one = monitor._ewma
    monitor.level()
    after_two = monitor._ewma

    assert 1.0 > after_one > after_two >= 0.0  # one decay step per sample, monotone


# --- join ordering must cost the orientation the plan will actually run -------------


def test_join_ordering_prices_the_orientation_selection_will_choose():
    """`hash_build_row` is twice `hash_probe_row`, so a join's cost depends on which side
    is built. The build-side rule runs in SELECTION, *after* JOIN_REORDER — so the reorder
    was ranking orders by an orientation the physical plan would then flip, penalizing
    every order that happened to put the large table on the right.

    `op_cost` stays orientation-specific (the build-side rule compares the two against
    each other); `join_op_cost` prices the cheaper one, which is what SELECTION picks.
    """
    import batcher as bt
    from batcher.kyber.cost import CostModel
    from batcher.kyber.stats import StatsEstimator

    big = bt.from_pydict({"k": list(range(1000)), "a": list(range(1000))})
    small = bt.from_pydict({"k": list(range(10)), "b": list(range(10))})
    cardinality = active_config().optimizer.cardinality

    def costs(dataset):
        model = CostModel(StatsEstimator(dataset._sources, {}, cardinality))
        plan = dataset._plan
        return model.op_cost(plan).total(), model.join_op_cost(plan).total()

    good_written, good_ordering = costs(big.join(small, on="k"))
    bad_written, bad_ordering = costs(small.join(big, on="k"))

    # `op_cost` must still discriminate, or the build-side rule cannot choose.
    assert bad_written > good_written
    # Join *ordering* sees one price for two commutative orientations.
    assert bad_ordering == pytest.approx(good_ordering)


def test_a_non_commutative_join_is_priced_as_written():
    import batcher as bt
    from batcher.kyber.cost import CostModel
    from batcher.kyber.stats import StatsEstimator

    big = bt.from_pydict({"k": list(range(1000)), "a": list(range(1000))})
    small = bt.from_pydict({"k": list(range(10)), "b": list(range(10))})
    ds = small.join(big, on="k", how="left")  # a LEFT join cannot swap its sides
    model = CostModel(StatsEstimator(ds._sources, {}, active_config().optimizer.cardinality))
    assert model.join_op_cost(ds._plan).total() == pytest.approx(model.op_cost(ds._plan).total())
