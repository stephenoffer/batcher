"""Learned execution-time sizing for the distributed executor.

Each decision reads a *measured* signal — the operator feedback Core already records, or a small
per-signature EMA the executor folds — and seeds a scheduling parameter: partition count, per-task
CPU share, shuffle fan-out, inference actor-pool size, straggler-speculation threshold. On a cold
store every reader returns `None` (or the caller's default), so a first run is unchanged.

None of these change a result — partition count only shards, CPU share only packs, reducer count
only rebalances buckets (correct under the mergeable algebra), speculation only duplicates-then-
dedupes — so the invariance checks here assert the *decision* moves with the signal while the
result-invariance is structural (proven by the distributed-equivalence suite).
"""

from __future__ import annotations

import pytest

from batcher.config import active_config
from batcher.dist.adaptive_sizing import sizing as az
from batcher.metadata import MetadataHub
from batcher.metadata.backends import InProcessBackend
from batcher.plan.feedback import OperatorFeedback
from batcher.plan.ids import OpId

pytestmark = pytest.mark.unit


def _hub() -> MetadataHub:
    return MetadataHub(InProcessBackend())


def _feedback(kind: str, *, cpu_util: float = 0.0, n_input: int = 0, t_ms: float = 0.0):
    return OperatorFeedback(
        op_id=OpId(1),
        kind=kind,
        n_actual=n_input,
        t_op_ms=t_ms,
        m_peak_bytes=0,
        selectivity=1.0,
        batch_size=16_384,
        cpu_utilization=cpu_util,
        n_input=n_input,
    )


# --- partition rows (EMA, folded by the executor) ----------------------------------------


def test_partition_rows_cold_then_learned():
    hub = _hub()
    assert az.learned_partition_rows(hub, "src") is None  # cold
    for _ in range(az._MIN_SAMPLES):
        az.record_partition_rows(hub, "src", 5_000_000)
    assert az.learned_partition_rows(hub, "src") == 5_000_000


def test_partition_rows_needs_min_samples():
    hub = _hub()
    az.record_partition_rows(hub, "src", 1_000)
    assert az.learned_partition_rows(hub, "src") is None  # one sample < the confidence gate


# --- CPU weight factor (from recorded cpu_utilization) -----------------------------------


def test_cpu_weight_factor_shrinks_when_underutilized():
    hub = _hub()
    for _ in range(az._MIN_SAMPLES):
        hub.record(_feedback("map_batches", cpu_util=0.2))  # cores mostly idle (IO/GPU-bound)
    factor = az.learned_cpu_weight_factor(hub, "map_batches")
    assert factor is not None and factor <= 0.3  # reserve far fewer cores next run


def test_cpu_weight_factor_full_when_saturated():
    hub = _hub()
    for _ in range(az._MIN_SAMPLES):
        hub.record(_feedback("aggregate", cpu_util=0.9))  # cores busy
    assert az.learned_cpu_weight_factor(hub, "aggregate") == pytest.approx(1.0)


def test_cpu_weight_factor_cold_is_none():
    assert az.learned_cpu_weight_factor(_hub(), "map_batches") is None


# --- shuffle fan-out (from recorded shuffle-family input rows) ----------------------------


def test_shuffle_fanout_trims_for_small_shuffle():
    hub = _hub()
    for _ in range(az._MIN_SAMPLES):
        hub.record(_feedback("aggregate", n_input=1_000))  # tiny shuffle
    assert az.learned_shuffle_fanout(hub, None, workers=64) == 1  # one full bucket suffices


def test_shuffle_fanout_keeps_fanout_for_large_shuffle():
    hub = _hub()
    big = active_config().optimizer.target_rows_per_task * 1000
    for _ in range(az._MIN_SAMPLES):
        hub.record(_feedback("hash_join", n_input=big))
    assert az.learned_shuffle_fanout(hub, "hash_join", workers=8) == 8  # keeps the full fan-out


def test_shuffle_fanout_cold_is_none():
    assert az.learned_shuffle_fanout(_hub(), None, workers=8) is None


# --- actor-pool size (EMA of served partitions) ------------------------------------------


def test_actor_pool_size_trims_to_served():
    hub = _hub()
    for _ in range(az._MIN_SAMPLES):
        az.record_actor_pool_reuse(hub, "sig", 2)
    assert az.learned_actor_pool_size(hub, "sig", default=10) == 2  # right-size to reuse
    assert az.learned_actor_pool_size(_hub(), "sig", default=10) is None  # cold


# --- straggler factor (from recorded task-time variance) ---------------------------------


def test_straggler_factor_high_for_uniform_times():
    hub = _hub()
    for _ in range(6):
        hub.record(_feedback("aggregate", t_ms=100.0))  # uniform → low variance
    default = active_config().distributed.speculation_straggler_factor
    factor = az.learned_straggler_factor(hub, "aggregate")
    assert factor is not None and factor > default  # rarely back up a tight distribution


def test_straggler_factor_lower_for_heavy_tail():
    """Below the *default*, not merely below the top of the band.

    `factor < default * 2.0` is the whole band's ceiling, so it held even while the formula
    could not produce anything under `default` at all — every family with a coefficient of
    variation at or above 1 was pinned exactly there, which made the loop's documented
    behaviour ("a heavy-tailed family backs up sooner") unreachable and the `0.75 x default`
    lower bound guard a region nothing could enter.
    """
    hub = _hub()
    for t in (1.0, 1.0, 1.0, 1.0, 1.0, 200.0):  # a fat tail
        hub.record(_feedback("sort", t_ms=t))
    default = active_config().distributed.speculation_straggler_factor
    factor = az.learned_straggler_factor(hub, "sort")
    assert factor is not None and factor < default  # a real straggler → back up sooner
    assert factor >= default * 0.75  # ...and no further than the band allows


def test_straggler_factor_cold_is_none():
    assert az.learned_straggler_factor(_hub(), "aggregate") is None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
