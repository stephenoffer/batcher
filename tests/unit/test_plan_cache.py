"""The optimizer's plan memo returns a plan only for the query it was built for.

Reusing an optimized plan is safe exactly as far as the key is: Kyber's rewrites are
semantics-preserving, so a stale plan is a slower plan — but a plan keyed by something that
*ignores* part of the query is a wrong answer. Two of those traps are live here:
`plan_signature` normalizes literals (so it must never key this cache), and an in-memory
source's `identity()` is only its shape (so two different relations collide on it, and
zone-map pruning reads a source's real bounds).

These tests pin the key, the invalidation, and the disable switch.
"""

from __future__ import annotations

import dataclasses

import pyarrow as pa
import pytest

import batcher as bt
from batcher.config import active_config, config_context
from batcher.io.source import InMemorySource
from batcher.kyber import learning, plan_cache

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_cache():
    plan_cache.clear()
    yield
    plan_cache.clear()


def _source(rows: list[int]) -> InMemorySource:
    return InMemorySource(pa.table({"x": pa.array(rows, type=pa.int64())}).to_batches())


def _plan_ir(predicate_literal: int) -> str:
    """The plan's content fingerprint (what `cache_key` now keys on) for `x > literal`."""
    table = pa.table({"x": pa.array([1, 2, 3], type=pa.int64())})
    return bt.from_arrow(table).filter(bt.col("x") > predicate_literal)._plan.content_key()


def _key(plan_key, sources, hub=None, config=None, kind="full"):
    return plan_cache.cache_key(plan_key, sources, config or active_config(), hub, kind)


# --- what the key must separate ----------------------------------------------


def test_a_different_literal_is_a_different_key():
    """The trap: `plan_signature` normalizes literals, so it could never key this cache.
    A plan built for `x > 1` must never be served to `x > 2`."""
    src = [_source([1, 2, 3])]
    assert _key(_plan_ir(1), src) != _key(_plan_ir(2), src)


def test_the_same_query_over_the_same_source_is_the_same_key():
    src = [_source([1, 2, 3])]
    assert _key(_plan_ir(1), src) == _key(_plan_ir(1), src)


def test_two_in_memory_sources_of_the_same_shape_do_not_collide():
    """`InMemorySource.identity()` is schema + row count, so different data shares it. Zone-map
    pruning reads the real bounds, so a shared plan could be a wrong answer, not a slow one."""
    a, b = _source([1, 2, 3]), _source([100, 200, 300])
    assert a.identity() == b.identity()  # the collision this key must survive
    assert _key(_plan_ir(1), [a]) != _key(_plan_ir(1), [b])


def test_a_different_optimizer_config_is_a_different_key():
    src = [_source([1, 2, 3])]
    base = active_config()
    tweaked = base.replace(optimizer=dataclasses.replace(base.optimizer, plan_cache_entries=8))
    assert _key(_plan_ir(1), src, config=base) != _key(_plan_ir(1), src, config=tweaked)


def test_a_different_hub_is_a_different_key():
    """Resetting the MetadataHub must invalidate every plan learned against the old one."""
    src = [_source([1, 2, 3])]
    assert _key(_plan_ir(1), src, hub=object()) != _key(_plan_ir(1), src, hub=object())


def test_the_two_optimizer_entry_points_do_not_collide():
    """`optimize_full` and `optimize_logical` are pure in the same inputs but return
    different shapes. Sharing the memo means the key must separate them, or a caller
    asking for a physical plan is handed a logical one."""
    src = [_source([1, 2, 3])]
    assert _key(_plan_ir(1), src, kind="full") != _key(_plan_ir(1), src, kind="logical")


def test_an_unkeyable_source_is_never_cached():
    class NoIdentity:
        pass

    assert _key(_plan_ir(1), [NoIdentity()]) is None


# --- invalidation -------------------------------------------------------------


def test_learning_something_plan_relevant_invalidates():
    """The generation advances only for a *material* correction, and the key follows it."""
    src = [_source([1, 2, 3])]
    before = _key(_plan_ir(1), src)
    learning._bump_generation()
    assert _key(_plan_ir(1), src) != before


def test_a_settled_estimate_does_not_invalidate():
    """A smoothed average drifting in its fourth decimal must not throw the plan away —
    that is why fingerprinting the stats' *content* never hits."""
    assert not learning._is_material(1000.0, 1001.0)  # 0.1%
    assert learning._is_material(1000.0, 1200.0)  # 20%
    assert learning._is_material(None, 5.0)  # nothing was known
    assert learning._is_material(0.0, 5.0)  # a provably-empty prior, now non-empty


# --- the store ----------------------------------------------------------------


def test_store_and_lookup_round_trip():
    src = [_source([1, 2, 3])]
    key = _key(_plan_ir(1), src)
    assert plan_cache.lookup(key) is None
    plan_cache.store(key, "plan", src, max_entries=4)
    assert plan_cache.lookup(key) == "plan"


def test_a_zero_cap_disables_the_cache():
    src = [_source([1, 2, 3])]
    key = _key(_plan_ir(1), src)
    plan_cache.store(key, "plan", src, max_entries=0)
    assert plan_cache.lookup(key) is None


def test_the_cap_evicts_least_recently_used():
    sources = [[_source([i])] for i in range(3)]
    keys = [_key(_plan_ir(1), s) for s in sources]
    for k, s in zip(keys, sources, strict=True):
        plan_cache.store(k, "plan", s, max_entries=2)
    assert plan_cache.lookup(keys[0]) is None  # evicted
    assert plan_cache.lookup(keys[1]) == "plan"
    assert plan_cache.lookup(keys[2]) == "plan"


def test_a_hit_refreshes_its_lru_position():
    sources = [[_source([i])] for i in range(3)]
    keys = [_key(_plan_ir(1), s) for s in sources]
    plan_cache.store(keys[0], "a", sources[0], max_entries=2)
    plan_cache.store(keys[1], "b", sources[1], max_entries=2)
    assert plan_cache.lookup(keys[0]) == "a"  # touch the oldest
    plan_cache.store(keys[2], "c", sources[2], max_entries=2)
    assert plan_cache.lookup(keys[1]) is None  # `b` was the least recently used
    assert plan_cache.lookup(keys[0]) == "a"


# --- end to end ---------------------------------------------------------------


def test_a_stage_boundary_source_reports_rows_without_scanning_for_bounds():
    """The adaptive executor wraps each stage's output in a throwaway source. Its exact
    row count is what re-optimization reads; the O(rows) min/max pass would be recomputed
    and discarded every run, so it must not happen."""
    from batcher.plan.stats import Provenance

    batches = pa.table({"x": pa.array([5, 1, 9], type=pa.int64())}).to_batches()
    ephemeral = InMemorySource(batches, zone_maps=False)
    assert ephemeral.statistics().row_count == 3
    assert ephemeral.statistics().columns == {}

    registered = InMemorySource(batches)  # a user table is queried again: bounds pay off
    stats = registered.statistics()
    assert stats.row_count == 3
    assert (stats.columns["x"].min, stats.columns["x"].max) == (1, 9)
    assert stats.columns["x"].provenance is Provenance.EXACT


def test_caching_never_changes_a_result_across_literals():
    """The whole point: run the same shape with different literals, cache on, and every
    answer must be the one that query asks for."""
    table = pa.table({"x": pa.array([1, 2, 3, 4, 5], type=pa.int64())})
    dataset = bt.from_arrow(table)
    for threshold in (0, 2, 4, 2, 0):
        got = dataset.filter(bt.col("x") > threshold).collect().to_pydict()["x"]
        assert got == [v for v in [1, 2, 3, 4, 5] if v > threshold]


def test_disabling_the_cache_gives_the_same_results():
    table = pa.table({"x": pa.array([1, 2, 3], type=pa.int64())})
    base = active_config()
    off = base.replace(optimizer=dataclasses.replace(base.optimizer, plan_cache_entries=0))
    with config_context(off):
        first = bt.from_arrow(table).filter(bt.col("x") > 1).collect().to_pydict()
    second = bt.from_arrow(table).filter(bt.col("x") > 1).collect().to_pydict()
    assert first == second == {"x": [2, 3]}


# --- a bandit arm must be able to invalidate a memoized plan ------------------


def test_a_bandit_arms_mean_invalidates_but_a_bare_tick_does_not():
    """`record_arm` writes only accumulators — `n`, `sum`, `sumsq` — all bookkeeping.

    With none of them compared, the key-set difference is empty and `any(())` is False, so the
    write could *never* bump the generation: the arm's ranking would move while a plan chosen
    under the old one was served forever. Comparing the raw counters instead would bump on
    every execution and defeat the memo. The *ratio* — the mean reward `ucb1_best_arm` ranks
    by — is what must decide.
    """
    from batcher.kyber.plan_cache import _materially_differs

    # A run that ticks `n` without moving the mean stays a cache hit.
    assert not _materially_differs({"sum": 100.0, "n": 10}, {"sum": 110.0, "n": 11})
    # A run that moves the mean materially must invalidate.
    assert _materially_differs({"sum": 100.0, "n": 10}, {"sum": 300.0, "n": 11})


def test_the_calibration_epoch_advances_only_with_a_refit():
    """The epoch is the identity of the live cost fit, not a bucket of the feedback counter.

    Those read as the same quantity and are different clocks. `calibrate`'s throttle counts
    feedback rows *since its last refit*; `version // _RECALIBRATE_AFTER` counts from zero. On
    TPC-H q8 at sf10, which records ~35 operators per execution, the bucket rolled over every
    second run over a completely stable set of coefficients — so the plan cache alternated
    hit/miss forever, and a miss costs 350 ms there against a hit's 160 ms.
    """
    import batcher as bt
    from batcher.config import active_config
    from batcher.kyber import calibration, cpu_shares, plan_cache
    from batcher.metadata import MetadataHub
    from batcher.metadata.backends import InProcessBackend

    hub = MetadataHub(InProcessBackend())
    ds = bt.from_pydict({"x": [1, 2, 3]})
    cfg, pk = active_config(), ds._plan.content_key()

    assert calibration.refit_version(hub) == 0  # nothing fitted yet
    assert cpu_shares.refit_version(hub) == 0
    first = plan_cache.cache_key(pk, ds._sources, cfg, hub)

    # Feedback accrues — the hub version climbs — but no refit has replaced either fit, so the
    # plan chosen under the old one is still the plan the new one would choose.
    calibration.calibrate(hub, cfg)
    before = calibration.refit_version(hub)
    for _ in range(200):
        hub.record(_feedback_row())
    assert plan_cache.cache_key(pk, ds._sources, cfg, hub) == plan_cache.cache_key(
        pk, ds._sources, cfg, hub
    )
    assert calibration.refit_version(hub) == before, "the fit changed without being re-run"

    # A refit past the throttle does advance it, so a plan chosen under the old coefficients
    # is not served under the new ones.
    calibration.calibrate(hub, cfg)
    assert calibration.refit_version(hub) != before
    assert plan_cache.cache_key(pk, ds._sources, cfg, hub) != first


def _feedback_row():
    from batcher.plan.feedback import OperatorFeedback
    from batcher.plan.ids import OpId

    return OperatorFeedback(
        op_id=OpId(0),
        kind="filter",
        n_actual=50,
        t_op_ms=1.0,
        m_peak_bytes=1024,
        selectivity=0.5,
        batch_size=16384,
        n_input=100,
    )


def test_source_stats_are_part_of_the_plan_cache_key():
    """Zone-map pruning folds a filter to FALSE from footer min/max carried in `source_stats`.

    Keying only on the source *object* let two calls with different collected statistics
    collide, serving the first call's pruned plan for the second — a wrong answer.
    """
    import batcher as bt
    from batcher.config import active_config
    from batcher.kyber import plan_cache
    from batcher.plan.source_stats import SourceStatistics
    from batcher.plan.stats import ColumnStat, Provenance

    ds = bt.from_pydict({"x": [1, 2, 3]})
    cfg, pk = active_config(), ds._plan.content_key()

    def stats(lo, hi):
        return [
            SourceStatistics(
                row_count=3,
                exact_rows=True,
                columns={"x": ColumnStat(min=lo, max=hi, provenance=Provenance.EXACT)},
            )
        ]

    none_key = plan_cache.cache_key(pk, ds._sources, cfg, None)
    low = plan_cache.cache_key(pk, ds._sources, cfg, None, source_stats=stats(1, 3))
    high = plan_cache.cache_key(pk, ds._sources, cfg, None, source_stats=stats(100, 300))
    assert none_key != low != high and none_key != high
    # Identical statistics must still key identically, or the memo never hits.
    assert low == plan_cache.cache_key(pk, ds._sources, cfg, None, source_stats=stats(1, 3))


# --- learned accumulators must not flush the cache on every single run -------------


def test_converged_ols_statistics_do_not_invalidate():
    """`_fold_ols` writes monotonically-growing sufficient statistics every observation.

    Comparing `sx`/`sy`/`sxx`/`sxy` raw made every join run look material and flushed the
    whole plan cache — the "6 hits in 8 identical runs became 0" regression, fixed for `n`
    but still live for the accumulators beside it. What a plan reads is the *fit*, which is a
    function of the per-observation moments.
    """
    from batcher.kyber.plan_cache import _materially_differs

    prior = {"n": 100, "sx": 1000.0, "sy": 2000.0, "sxx": 12000.0, "sxy": 21000.0,
             "xmin": 1.0, "xmax": 50.0}  # fmt: skip
    one_more = {"n": 101, "sx": 1010.0, "sy": 2020.0, "sxx": 12120.0, "sxy": 21210.0,
                "xmin": 1.0, "xmax": 50.0}  # fmt: skip
    assert not _materially_differs(prior, one_more)


def test_a_shifted_ols_relationship_still_invalidates():
    from batcher.kyber.plan_cache import _materially_differs

    prior = {"n": 100, "sx": 1000.0, "sy": 2000.0, "sxx": 12000.0, "sxy": 21000.0,
             "xmin": 1.0, "xmax": 50.0}  # fmt: skip
    shifted = {"n": 101, "sx": 1010.0, "sy": 4000.0, "sxx": 12120.0, "sxy": 45000.0,
               "xmin": 1.0, "xmax": 50.0}  # fmt: skip
    assert _materially_differs(prior, shifted)
    # A genuinely new x extreme changes the fit's applicable range, so it invalidates too.
    extreme = dict(prior, n=101, xmax=500.0)
    assert _materially_differs(prior, extreme)


def test_bandit_arm_invalidates_on_its_mean_not_its_accumulator():
    """`record_arm` grows `sum`/`sumsq` every run; `ucb1_best_arm` ranks by `sum/n`."""
    from batcher.kyber.plan_cache import _materially_differs

    prior = {"hash": {"n": 50, "sum": 500.0, "sumsq": 6000.0}}
    stable = {"hash": {"n": 51, "sum": 510.0, "sumsq": 6120.0}}  # same ~10ms mean
    doubled = {"hash": {"n": 51, "sum": 1020.0, "sumsq": 24000.0}}  # mean ~20ms
    assert not _materially_differs(prior, stable)
    assert _materially_differs(prior, doubled)
