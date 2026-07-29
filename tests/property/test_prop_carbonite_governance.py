"""Property: Carbonite's sizing rules are monotone, bounded, and never degenerate.

Every decision this subsystem makes is result-invariant, which is exactly why its bugs
survive: the differential suite compares *answers*, and no answer changes when a credit
window is ten times too large or a spill is sharded into two buckets that each still
exceed the budget. What can be checked is the shape of the rules themselves, and the
useful properties are monotonicity and boundedness:

- more bytes to shard must never mean fewer buckets;
- a fuller box must never classify as calmer;
- a credit window must stay inside its band no matter what sequence of congestion
  signals the channel sees;
- an eviction must free at least what it was asked for, or empty the cache trying.

Each of these is the kind of thing that holds for the handful of values a unit test
picks and fails on the eleventh. Hypothesis picks the eleventh.
"""

from __future__ import annotations

import pyarrow as pa
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from batcher.carbonite.cache import CacheStore
from batcher.carbonite.memory.learned import _upper_quantile
from batcher.carbonite.memory.pool import BufferPool
from batcher.carbonite.memory.pressure import PressureLevel, PressureMonitor
from batcher.carbonite.policies.concurrency import ConcurrencyLimiter
from batcher.carbonite.policies.flow_control import AIMDFlowControl, credit_ceiling
from batcher.carbonite.policies.morsel import MIN_MORSEL_BYTES, MIN_MORSEL_ROWS, morsel_target
from batcher.carbonite.policies.spill_shape import (
    MAX_SPILL_PARTITIONS,
    MIN_SPILL_PARTITIONS,
    partitions_for_envelope,
    partitions_for_volume,
    spill_basis,
)
from batcher.config import Config, FlowControlConfig, MemoryConfig

pytestmark = [pytest.mark.property, pytest.mark.unit]

_SLOW = settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])


# --- spill shape -------------------------------------------------------------


@given(basis=st.integers(min_value=1, max_value=1 << 50))
@_SLOW
def test_partition_count_stays_inside_its_band(basis: int) -> None:
    parts = partitions_for_volume(basis)
    assert parts is not None
    assert MIN_SPILL_PARTITIONS <= parts <= MAX_SPILL_PARTITIONS


@given(
    a=st.integers(min_value=1, max_value=1 << 45),
    b=st.integers(min_value=1, max_value=1 << 45),
)
@_SLOW
def test_more_bytes_never_means_fewer_buckets(a: int, b: int) -> None:
    """Monotone in the basis: a bigger spilled state cannot shard into less."""
    lo, hi = sorted((a, b))
    assert (partitions_for_volume(hi) or 0) >= (partitions_for_volume(lo) or 0)


@given(
    basis=st.integers(min_value=1, max_value=1 << 45),
    envelope=st.integers(min_value=1, max_value=1 << 40),
)
@_SLOW
def test_the_counter_offer_is_honoured_or_the_cap_is_reached(basis: int, envelope: int) -> None:
    """Each bucket fits the offered envelope, unless the bucket cap binds first."""
    parts = partitions_for_envelope(basis, envelope)
    assert parts >= 1
    if parts < MAX_SPILL_PARTITIONS:
        assert -(-basis // parts) <= envelope, "a bucket still exceeds the offered envelope"


@given(
    peak=st.integers(min_value=0, max_value=1 << 45),
    volume=st.integers(min_value=0, max_value=1 << 45),
)
@_SLOW
def test_the_spill_basis_is_never_negative_and_prefers_the_measurement(
    peak: int, volume: int
) -> None:
    basis = spill_basis(peak, volume)
    assert basis >= 0
    assert basis == (volume if volume > 0 else max(0, peak))


# --- pressure ----------------------------------------------------------------


@given(
    used=st.floats(min_value=0.0, max_value=1.5, allow_nan=False),
    soft=st.floats(min_value=0.05, max_value=0.99),
    hard=st.floats(min_value=0.05, max_value=1.0),
)
@_SLOW
def test_pressure_is_monotone_for_any_config(used: float, soft: float, hard: float) -> None:
    """A fuller box never classifies as calmer, whatever the soft/hard pair is."""
    mon = PressureMonitor(Config().replace(memory=MemoryConfig(soft_limit=soft, hard_limit=hard)))
    here = mon._classify(used)
    assert here in set(PressureLevel)
    assert mon._classify(min(1.5, used + 0.05)) >= here


# --- flow control ------------------------------------------------------------


@given(
    signals=st.lists(st.booleans(), min_size=1, max_size=120),
    credits=st.integers(min_value=1, max_value=64),
)
@_SLOW
def test_the_credit_window_never_leaves_its_band(signals: list[bool], credits: int) -> None:
    """No sequence of congestion signals can take the window outside `[1, ceiling]`."""
    cfg = Config().replace(flow_control=FlowControlConfig(default_credits=credits))
    ceiling = credit_ceiling(cfg)
    ctrl = AIMDFlowControl(cfg)
    for congested in signals:
        window = ctrl.observe(congested=congested)
        assert 1 <= window <= ceiling


@given(signals=st.lists(st.booleans(), min_size=1, max_size=60))
@_SLOW
def test_congestion_never_grows_the_window(signals: list[bool]) -> None:
    """The multiplicative decrease must actually decrease — an unstable law otherwise."""
    ctrl = AIMDFlowControl(Config())
    for congested in signals:
        before = ctrl.window
        after = ctrl.observe(congested=congested)
        if congested:
            assert after <= before


@given(channels=st.integers(min_value=1, max_value=512))
@_SLOW
def test_more_channels_never_get_a_larger_per_channel_ceiling(channels: int) -> None:
    """The whole-shuffle byte budget is shared, so per-channel share is non-increasing."""
    cfg = Config()
    assert credit_ceiling(cfg, channels=channels) >= 1
    assert credit_ceiling(cfg, channels=channels * 2) <= credit_ceiling(cfg, channels=channels)


# --- morsel sizing -----------------------------------------------------------


@given(level=st.sampled_from(list(PressureLevel)))
@_SLOW
def test_a_morsel_never_shrinks_below_its_floors(level: PressureLevel) -> None:
    target = morsel_target(Config(), level)
    if target is None:
        assert level is PressureLevel.NORMAL
        return
    rows, nbytes = target
    assert rows >= MIN_MORSEL_ROWS
    assert nbytes >= MIN_MORSEL_BYTES
    assert rows <= Config().execution.morsel_rows
    assert nbytes <= Config().execution.morsel_bytes


# --- the buffer pool ---------------------------------------------------------


@given(
    limit=st.integers(min_value=1, max_value=1 << 30),
    requests=st.lists(st.integers(min_value=-1000, max_value=1 << 20), min_size=1, max_size=40),
)
@_SLOW
def test_the_pool_never_admits_past_its_limit(limit: int, requests: list[int]) -> None:
    """Reserve-before-allocate is only real if `used` can never exceed `limit`."""
    pool = BufferPool(limit)
    import contextlib

    with contextlib.ExitStack() as stack:
        for n in requests:
            stack.enter_context(pool.reserve(n))
            assert 0 <= pool.used <= limit
            assert 0.0 <= pool.utilization <= 1.0
    assert pool.used == 0, "every reservation is released on scope exit"


# --- the result cache --------------------------------------------------------


@given(
    budget=st.integers(min_value=0, max_value=1 << 16),
    sizes=st.lists(st.integers(min_value=1, max_value=2000), min_size=1, max_size=25),
)
@_SLOW
def test_the_cache_never_exceeds_its_budget(budget: int, sizes: list[int]) -> None:
    store = CacheStore(budget)
    for i, n in enumerate(sizes):
        store.put(f"k{i}", pa.table({"v": pa.array(range(n), type=pa.int64())}))
        assert store.used_bytes <= store.max_bytes
    stats = store.stats()
    assert stats["used_bytes"] == store.used_bytes
    assert stats["entries"] == len(store)


@given(
    sizes=st.lists(st.integers(min_value=1, max_value=500), min_size=1, max_size=20),
    want=st.integers(min_value=1, max_value=1 << 20),
)
@_SLOW
def test_evict_to_free_frees_the_deficit_or_empties_the_cache(sizes: list[int], want: int) -> None:
    store = CacheStore(1 << 24)
    for i, n in enumerate(sizes):
        store.put(f"k{i}", pa.table({"v": pa.array(range(n), type=pa.int64())}))
    freed = store.evict_to_free(want)
    assert freed >= want or store.used_bytes == 0


# --- concurrency -------------------------------------------------------------


@given(
    slots=st.integers(min_value=1, max_value=64),
    active=st.integers(min_value=1, max_value=256),
    cores=st.integers(min_value=1, max_value=256),
)
@_SLOW
def test_the_pool_width_never_exceeds_the_cores(slots: int, active: int, cores: int) -> None:
    """N queries each asking for every core is the oversubscription this exists to stop."""
    limiter = ConcurrencyLimiter(slots=slots, cores=cores)
    width = limiter.width_for(active)
    assert width >= 0
    if active > 1:
        assert 1 <= width <= cores
        assert width * active <= cores + active  # at most one core of rounding slack each


# --- the learned fit ---------------------------------------------------------


@given(
    values=st.lists(
        st.floats(min_value=0.001, max_value=1e9, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=60,
    )
)
@_SLOW
def test_the_quantile_lies_between_the_extremes_and_above_the_median(
    values: list[float],
) -> None:
    import statistics

    q = _upper_quantile(values)
    assert min(values) <= q <= max(values), "the quantile left the sample's own range"
    # Relative tolerance on the median comparison only: `_upper_quantile` is clamped to
    # the order statistics it interpolates, but `statistics.median` of an even-length
    # sample is itself a float mean, so the two can differ by an ulp on a constant sample.
    median = statistics.median(values)
    assert q >= median - abs(median) * 1e-9 - 1e-12, (
        "an upper quantile must not sit below the median"
    )
