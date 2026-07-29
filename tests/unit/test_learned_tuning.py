"""Unit tests for Kyber's learned strategy + parameter tuning (`kyber.learned_tuning`).

Each decision is checked twice: **cold** (an empty hub → the function returns `None`/default, so
a first run keeps the static behavior) and **warm** (seeded measured signals → the decision moves
to the learned value). Result-invariance is proven separately in the differential suite; these
tests pin the *decision*, which is what changes plan shape.
"""

from __future__ import annotations

import pytest

from batcher.kyber import learned_tuning as lt
from batcher.metadata import MetadataHub
from batcher.metadata.backends import InProcessBackend

pytestmark = pytest.mark.unit


def _hub() -> MetadataHub:
    return MetadataHub(InProcessBackend())


# --- the UCB1 bandit primitive ----------------------------------------------------------------
def test_ucb1_returns_none_when_no_arm_tried():
    assert lt.ucb1_best_arm({}, ("hash", "broadcast", "sort_merge")) is None


def test_ucb1_explores_an_untried_arm_first():
    # Two arms sampled, one untried → the untried arm is explored (deterministically, lowest name).
    stats = {"hash": {"n": 3, "mean": 10, "m2": 0.0}, "sort_merge": {"n": 3, "mean": 3, "m2": 0.0}}
    got = lt.ucb1_best_arm(stats, ("broadcast", "hash", "sort_merge"))
    assert got == "broadcast"


def test_ucb1_exploits_the_fastest_arm_once_all_tried():
    # All arms tried; sort_merge is clearly fastest (lowest mean latency) → it wins.
    stats = {
        "hash": {"n": 5, "mean": 100, "m2": 0.0},  # mean 100 ms
        "broadcast": {"n": 5, "mean": 80, "m2": 0.0},  # mean 80 ms
        "sort_merge": {"n": 5, "mean": 10, "m2": 0.0},  # mean 10 ms
    }
    assert lt.ucb1_best_arm(stats, ("hash", "broadcast", "sort_merge")) == "sort_merge"


def test_ucb1_is_deterministic():
    stats = {
        "hash": {"n": 5, "mean": 100, "m2": 0.0},
        "broadcast": {"n": 5, "mean": 80, "m2": 0.0},
        "sort_merge": {"n": 5, "mean": 10, "m2": 0.0},
    }
    arms = ("hash", "broadcast", "sort_merge")
    assert len({lt.ucb1_best_arm(stats, arms) for _ in range(20)}) == 1


# --- learned join strategy (the bandit, keyed by signature) -----------------------------------
def test_learned_join_strategy_cold_is_none():
    assert lt.learned_join_strategy(_hub(), "sig-A") is None


def test_learned_join_strategy_warms_to_the_fastest_arm():
    hub = _hub()
    # Below the warm-up floor → still defer to the cost model.
    lt.record_join_strategy(hub, "sig-A", "hash", 100.0)
    assert lt.learned_join_strategy(hub, "sig-A") is None
    # Feed clear evidence that sort_merge is fastest for this signature.
    for _ in range(4):
        lt.record_join_strategy(hub, "sig-A", "hash", 100.0)
        lt.record_join_strategy(hub, "sig-A", "broadcast", 80.0)
        lt.record_join_strategy(hub, "sig-A", "sort_merge", 10.0)
    assert lt.learned_join_strategy(hub, "sig-A") == "sort_merge"
    # A different signature is still cold — learning is per-shape.
    assert lt.learned_join_strategy(hub, "sig-B") is None


# --- learned broadcast threshold (OLS crossover) ----------------------------------------------
_BYTES = (1e5, 5e5, 1e6, 3e6, 6e6, 1e7, 2e7, 4e7)


def test_learned_broadcast_threshold_cold_is_none():
    assert lt.learned_broadcast_max_bytes(_hub(), default=10 * 1024 * 1024) is None


def test_learned_broadcast_threshold_recovers_the_crossover():
    hub = _hub()
    # broadcast: cheap fixed cost but grows fast with build bytes (replication);
    # shuffle: dear fixed cost, flat. They cross near 8 MB.
    default = 10 * 1024 * 1024
    for b in _BYTES:
        lt.record_broadcast_timing(hub, "broadcast", b, 5.0 + 2.0e-6 * b)
        lt.record_broadcast_timing(hub, "shuffle", b, 25.0 + 0.5e-6 * b)
    got = lt.learned_broadcast_max_bytes(hub, default=default)
    assert got is not None
    true_xover = (25.0 - 5.0) / (2.0e-6 - 0.5e-6)  # ≈ 13.3 MB, inside the band around 10 MB
    assert abs(got - true_xover) / true_xover < 0.15


def test_learned_broadcast_no_crossover_when_broadcast_never_wins():
    hub = _hub()
    for b in _BYTES:
        lt.record_broadcast_timing(hub, "broadcast", b, 50.0 + 2.0e-6 * b)  # dearer everywhere
        lt.record_broadcast_timing(hub, "shuffle", b, 5.0 + 0.5e-6 * b)
    assert lt.learned_broadcast_max_bytes(hub, default=10 * 1024 * 1024) is None


# --- learned sort-merge crossover (OLS) -------------------------------------------------------
_ROWS = (1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8, 5e8)


def test_learned_sort_merge_cold_is_none():
    assert lt.learned_sort_merge_min_rows(_hub(), default=50_000_000.0) is None


def test_learned_sort_merge_recovers_the_crossover():
    hub = _hub()
    default = 50_000_000.0
    # hash: cheap fixed cost, steep per-row (memory thrash); sort_merge: dear fixed, flat.
    for r in _ROWS:
        lt.record_sort_merge_timing(hub, "hash", r, 10.0 + 2.0e-6 * r)
        lt.record_sort_merge_timing(hub, "sort_merge", r, 60.0 + 0.5e-6 * r)
    got = lt.learned_sort_merge_min_rows(hub, default=default)
    assert got is not None
    true_xover = (60.0 - 10.0) / (2.0e-6 - 0.5e-6)  # ≈ 33.3M, inside the band around 50M
    assert abs(got - true_xover) / true_xover < 0.15


# --- learned build-side prior -----------------------------------------------------------------
def test_learned_build_sides_cold_is_none():
    assert lt.learned_build_sides(_hub(), "j1") is None


def test_learned_build_sides_returns_measured_sizes():
    hub = _hub()
    lt.record_join_sides(hub, "j1", 6_000_000.0, 1_500_000.0)
    left, right = lt.learned_build_sides(hub, "j1")
    assert (round(left), round(right)) == (6_000_000, 1_500_000)


# --- learned partition/parallelism prior ------------------------------------------------------
def test_learned_partition_count_cold_is_none():
    assert lt.learned_partition_count(_hub(), "b1", target_rows=4_000_000) is None


def test_learned_partition_count_from_measured_rows():
    hub = _hub()
    lt.record_partition_rows(hub, "b1", 20_000_000.0)
    assert lt.learned_partition_count(hub, "b1", target_rows=4_000_000) == 5  # ceil(20M / 4M)


# --- learned partial-aggregation decision -----------------------------------------------------
def test_learned_partial_agg_cold_is_none():
    assert lt.learned_partial_agg(_hub(), "a1") is None


def test_learned_partial_agg_engages_on_high_reduction():
    hub = _hub()
    # 10M rows collapse to 100 groups → ratio ~0 → pre-aggregation pays off (engage).
    lt.record_group_reduction(hub, "a1", groups=100.0, input_rows=10_000_000.0)
    assert lt.learned_partial_agg(hub, "a1") is True


def test_learned_partial_agg_skips_when_no_reduction():
    hub = _hub()
    # Nearly every row is its own group → ratio ~1 → pre-aggregation is wasted (skip).
    lt.record_group_reduction(hub, "a2", groups=9_900_000.0, input_rows=10_000_000.0)
    assert lt.learned_partial_agg(hub, "a2") is False


# --- learned selectivity-primed intermediate estimate -----------------------------------------
def test_learned_signature_rows_reads_the_feedback_loop():
    import pyarrow as pa

    from batcher.kyber.learning import record_execution
    from batcher.plan.logical import Scan
    from batcher.plan.schema import SchemaRef

    hub = _hub()
    plan = Scan(0, SchemaRef(pa.schema([pa.field("k", pa.int64())])))
    from batcher.kyber.signature import plan_signature

    sig = plan_signature(plan)
    assert lt.learned_signature_rows(hub, sig) is None  # cold
    record_execution(hub, plan, output_rows=4242)
    assert lt.learned_signature_rows(hub, sig) == 4242.0


# --- learned adaptive routing (staged vs one-shot, a two-arm bandit) --------------------------
def test_adaptive_route_is_unknown_cold():
    """Nothing measured ⇒ no route, so the caller's structural heuristic decides."""
    assert lt.learned_adaptive_route(_hub(), "q1") is None


def test_adaptive_route_explores_the_untried_arm():
    """One arm measured is not evidence; the bandit must try the other before ranking."""
    hub = _hub()
    lt.record_adaptive_route(hub, "q1", "staged", 9999.0)  # cold sample, discarded
    lt.record_adaptive_route(hub, "q1", "staged", 500.0)
    assert lt.learned_adaptive_route(hub, "q1") == "one_shot"


def test_the_first_observation_of_each_route_is_discarded():
    """A shape's first run on a route pays one-time costs that recur for neither route.

    q18 at sf10 is 8,000 ms on its first execution and ~300 ms thereafter; whichever arm ran
    first would otherwise carry that 25x forever and the bandit would be ranking the order
    the arms were tried in.
    """
    hub = _hub()
    lt.record_adaptive_route(hub, "q", "staged", 8000.0)
    assert lt.learned_adaptive_route(hub, "q") is None, "the cold sample was recorded"

    lt.record_adaptive_route(hub, "q", "staged", 300.0)
    lt.record_adaptive_route(hub, "q", "one_shot", 8000.0)  # its own cold sample, discarded
    lt.record_adaptive_route(hub, "q", "one_shot", 900.0)
    assert lt.learned_adaptive_route(hub, "q") == "staged"


def test_adaptive_route_converges_on_the_faster_arm():
    """TPC-H q8 at sf10: 887 ms staged against 142 ms one-shot."""
    hub = _hub()
    for _ in range(5):
        lt.record_adaptive_route(hub, "q8", "staged", 887.0)
        lt.record_adaptive_route(hub, "q8", "one_shot", 142.0)
    assert lt.learned_adaptive_route(hub, "q8") == "one_shot"


def test_adaptive_route_keeps_staging_where_one_shot_is_worse():
    """The other direction, which a "staging never pays" verdict would get wrong.

    Which route wins is not a constant of the plan — staging is the only distributed route
    for some shapes, and it is what earns the statistics a cold shape has not learned yet.
    The router has to be able to say "staged" as readily as "one_shot".
    """
    hub = _hub()
    for _ in range(5):
        lt.record_adaptive_route(hub, "q_star", "staged", 169.0)
        lt.record_adaptive_route(hub, "q_star", "one_shot", 2500.0)
    assert lt.learned_adaptive_route(hub, "q_star") == "staged"


def test_adaptive_route_is_per_signature():
    hub = _hub()
    for _ in range(5):
        lt.record_adaptive_route(hub, "q8", "staged", 887.0)
        lt.record_adaptive_route(hub, "q8", "one_shot", 142.0)
    assert lt.learned_adaptive_route(hub, "q_other") is None


# --- confidence-gated, smoothed CPU-share model (cpu_shares refinement) ------------------------
def _record_filter_util(hub: MetadataHub, utils: list[float]) -> None:
    from batcher.plan.feedback import OperatorFeedback
    from batcher.plan.ids import OpId

    for u in utils:
        hub.record(
            OperatorFeedback(
                op_id=OpId(0),
                kind="filter",
                n_actual=1,
                t_op_ms=1.0,
                m_peak_bytes=0,
                selectivity=1.0,
                batch_size=1,
                cpu_utilization=u,
            )
        )


def test_cpu_confidence_gate_suppresses_a_dispersed_family():
    from batcher.config import active_config
    from batcher.kyber.cpu_shares import load_cpu_utilization

    n = active_config().optimizer.cost_calibration_min_samples
    # A wildly bimodal family (alternating 0.1 / 0.95) — its mean predicts neither mode, so the
    # confidence gate suppresses it and the static prior stands (tag absent from the map).
    dispersed = _hub()
    _record_filter_util(dispersed, [0.1 if i % 2 else 0.95 for i in range(n)])
    assert "filter" not in load_cpu_utilization(dispersed)
    # A concentrated family (all ~0.9) clears the gate and is emitted.
    tight = _hub()
    _record_filter_util(tight, [0.9] * n)
    assert load_cpu_utilization(tight)["filter"] == 0.9


# --- the learned join-strategy arm actually reaches the optimized plan -------------------------
def _find_join(plan):
    from batcher.plan.logical import Join
    from batcher.plan.visitor import walk

    return next(n for n in walk(plan) if isinstance(n, Join))


def test_learned_join_arm_overrides_the_cost_model_choice():
    import batcher as bt
    from batcher.kyber import optimize_full

    left = bt.from_pydict({"k": [1, 2, 3], "v": [1, 2, 3]})
    right = bt.from_pydict({"k": [1, 2], "w": [9, 8]})
    q = left.join(right, on="k")

    hub = _hub()
    # Cold: the SELECTION rule picks a strategy purely from the cost model.
    _p, logical, decisions = optimize_full(q._plan, sources=q._sources, hub=hub)
    join = _find_join(logical)
    # Key the reward on the signature SELECTION *looked the arm up under*, exactly as the
    # conductor does. It is not `plan_signature` of the finished plan: the ENFORCE phase runs
    # after SELECTION and rewrites the join's inputs (a runtime filter lands on them), which
    # changes the join's structural signature. Recording under the finished plan's signature
    # would teach an arm the optimizer never consults — the loop would silently never close.
    sig = decisions[0].signature
    assert sig  # SELECTION stamped the key it used
    cold_strategy = join.strategy
    assert cold_strategy != "sort_merge"

    # Teach the bandit that a *different admissible* arm is fastest for this signature.
    # Sort-merge is deliberately not used here: this join's build fits memory many times
    # over, so `selection._admissible_arms` withholds it from the bandit entirely (it
    # cannot win where memory is not binding, and exploring it costs a measured 10x — see
    # `test_sort_merge_is_not_offered_when_the_build_fits_memory`).
    loser = "broadcast" if cold_strategy == "hash" else "hash"
    for _ in range(4):
        lt.record_join_strategy(hub, sig, cold_strategy, 100.0)
        lt.record_join_strategy(hub, sig, loser, 5.0)

    _p2, logical2, _d2 = optimize_full(q._plan, sources=q._sources, hub=hub)
    assert _find_join(logical2).strategy == loser  # learned arm overrode the cost guess


def test_sort_merge_is_not_offered_when_the_build_fits_memory():
    """Sort-merge exists to bound memory; where memory is not binding it cannot win.

    The bandit gives every untried arm a turn and its evidence expires, so an inadmissible
    arm is re-explored roughly every `1/(1-discount)` runs. On TPC-H sf10
    `lineitem ⋈ orders` that experiment costs **12.8 s at 2.5x parallelism against 1.2 s at
    21x** — regret UCB cannot recover, because the arm was never a candidate.
    """
    from batcher.kyber.rules.selection import BuildSideDecision, _admissible_arms

    fits = BuildSideDecision(1e9, 1e9, False, "exact", build_bytes=1e9, build_measured=True)
    assert "sort_merge" not in _admissible_arms(fits, sort_merge_min_bytes=30e9)

    strains = BuildSideDecision(1e9, 1e9, False, "exact", build_bytes=90e9, build_measured=True)
    assert "sort_merge" in _admissible_arms(strains, sort_merge_min_bytes=30e9)


def test_an_unbounded_build_size_never_stands_down_the_memory_guard():
    """TPC-H q5: relaxing the gate on a join-output build size OOM-killed it at 134 GB.

    A join-free build is bounded by its scan's EXACT row count, whatever the intervening
    selectivities do. A build that is itself a join output has no such ceiling — its size is a
    product of guesses — so the byte test must not speak for it.
    """
    from batcher.kyber.rules.selection import BuildSideDecision, _admissible_arms

    guessed = BuildSideDecision(1e9, 1e9, False, "default", build_bytes=1e9, build_measured=False)
    assert "sort_merge" in _admissible_arms(guessed, sort_merge_min_bytes=30e9)


def test_both_hashing_arms_are_always_offered():
    """Narrowing the arm set must never leave the bandit with nothing to compare."""
    from batcher.kyber.rules.selection import BuildSideDecision, _admissible_arms

    arms = _admissible_arms(
        BuildSideDecision(1e9, 1e9, False, "exact", build_bytes=1.0, build_measured=True),
        sort_merge_min_bytes=30e9,
    )
    assert set(arms) == {"hash", "broadcast"}


def test_a_memory_fitting_build_is_not_sort_merged_cold():
    """The cost model's own cold choice, not just the bandit's exploration.

    A row floor alone said nothing about the machine or the row width: on TPC-H sf10 the
    build landed on the 57M-row `lineitem` side, cleared the 50M floor, and the resulting
    sort-merge join ran 10.4 s at 2.3x parallelism where hash ran 1.5 s at 20x — on a host
    where that build is under a gigabyte.
    """
    import batcher as bt
    from batcher.kyber.cardinality import CardinalityEstimator
    from batcher.kyber.rules.selection import adaptive_build_side

    left = bt.from_pydict({"k": list(range(64)), "v": list(range(64))})
    right = bt.from_pydict({"k": list(range(32)), "w": list(range(32))})
    q = left.join(right, on="k")
    est = CardinalityEstimator(q._sources, {})

    # A row floor of zero would sort-merge everything; the byte floor withholds it because
    # this build is tiny next to the stated memory.
    _plan, decisions = adaptive_build_side(
        q._plan,
        est,
        sort_merge_min_rows=0.0,
        sort_merge_min_bytes=1 << 30,
        broadcast_max_bytes=0,
    )
    assert decisions
    from batcher.plan.logical import Join
    from batcher.plan.visitor import walk

    assert all(n.strategy != "sort_merge" for n in walk(_plan) if isinstance(n, Join))


def test_a_join_output_build_is_not_treated_as_bounded():
    """The structural half of the q5 guard, on a real plan rather than a hand-made decision."""
    import batcher as bt
    from batcher.kyber.cardinality import CardinalityEstimator
    from batcher.kyber.rules.selection import adaptive_build_side

    a = bt.from_pydict({"k": list(range(64)), "v": list(range(64))})
    b = bt.from_pydict({"k": list(range(64)), "w": list(range(64))})
    c = bt.from_pydict({"k": list(range(64)), "z": list(range(64))})
    q = a.join(b.join(c, on="k"), on="k")  # the outer join's build side *is* a join output
    est = CardinalityEstimator(q._sources, {})

    _plan, decisions = adaptive_build_side(
        q._plan, est, sort_merge_min_rows=0.0, sort_merge_min_bytes=1 << 30, broadcast_max_bytes=0
    )
    assert any(not d.build_measured for d in decisions), (
        "no join reported an unbounded build, so the q5 guard could never engage"
    )


# --- all decisions are best-effort on a None hub ----------------------------------------------
def test_none_hub_is_safe_everywhere():
    assert lt.learned_join_strategy(None, "x") is None
    assert lt.learned_broadcast_max_bytes(None) is None
    assert lt.learned_sort_merge_min_rows(None, 1.0) is None
    assert lt.learned_build_sides(None, "x") is None
    assert lt.learned_partition_count(None, "x", 1) is None
    assert lt.learned_partial_agg(None, "x") is None
    assert lt.learned_signature_rows(None, "x") is None
    assert lt.learned_adaptive_route(None, "x") is None
    # record_* on a None hub is a no-op, never raises.
    lt.record_join_strategy(None, "x", "hash", 1.0)
    lt.record_broadcast_timing(None, "broadcast", 1.0, 1.0)
    lt.record_sort_merge_timing(None, "hash", 1.0, 1.0)
    lt.record_join_sides(None, "x", 1.0, 1.0)
    lt.record_partition_rows(None, "x", 1.0)
    lt.record_group_reduction(None, "x", 1.0, 1.0)
    lt.record_adaptive_route(None, "x", "staged", 1.0)
