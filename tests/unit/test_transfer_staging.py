"""Host-side staging: chunk size, pipeline depth, pinning, and what the budget shrinks.

The direction that matters is which knob gives way under a memory budget. Shrinking the chunk
below the link's bandwidth-delay product makes the transfer slower than not pipelining at all,
so the depth is what has to yield.
"""

from __future__ import annotations

import pytest

from batcher.carbonite.transfer.staging import (
    MAX_CHUNK_BYTES,
    MIN_CHUNK_BYTES,
    StagingPlan,
    chunk_bytes_for_link,
    effective_gbps,
    pinned_budget_bytes,
    pipeline_depth,
    plan_staging,
    staging_seconds,
    worth_pinning,
)

GB = 1024**3


def test_the_chunk_is_the_links_bandwidth_delay_product() -> None:
    """100 GB/s across 10 us is 1 MB in flight, rounded down to a power of two."""
    assert chunk_bytes_for_link(100.0, latency_us=10.0) == 1 << 19


def test_a_product_below_the_floor_is_raised_to_it() -> None:
    """25 GB/s across 10 us is 250 KB, under the smallest chunk worth issuing."""
    assert chunk_bytes_for_link(25.0, latency_us=10.0) == MIN_CHUNK_BYTES


def test_the_chunk_is_a_power_of_two() -> None:
    for rate in (5.0, 25.0, 64.0, 400.0):
        chunk = chunk_bytes_for_link(rate)
        assert chunk & (chunk - 1) == 0


def test_an_unknown_link_gets_the_small_safe_chunk() -> None:
    assert chunk_bytes_for_link(0.0) == MIN_CHUNK_BYTES
    assert chunk_bytes_for_link(25.0, latency_us=0.0) == MIN_CHUNK_BYTES


def test_the_chunk_is_clamped_at_both_ends() -> None:
    assert chunk_bytes_for_link(0.001) == MIN_CHUNK_BYTES
    assert chunk_bytes_for_link(100_000.0, latency_us=1000.0) == MAX_CHUNK_BYTES


def test_a_known_link_double_buffers_at_minimum() -> None:
    assert pipeline_depth(1 << 20, 25.0) == 2


def test_a_slow_producer_deepens_the_ring_until_the_link_stops_waiting() -> None:
    chunk = 1 << 20
    copy_us = chunk / 25e9 * 1e6
    assert pipeline_depth(chunk, 25.0, prepare_us=copy_us * 3) == 4


def test_the_ring_is_capped_so_it_cannot_hold_the_machine() -> None:
    assert pipeline_depth(1 << 20, 25.0, prepare_us=1e6) == 8


def test_an_unknown_link_does_not_pipeline() -> None:
    assert pipeline_depth(1 << 20, 0.0) == 1
    assert pipeline_depth(0, 25.0) == 1


def test_pinning_pays_above_a_megabyte_and_not_below() -> None:
    assert worth_pinning(4 * 1024 * 1024)
    assert not worth_pinning(64 * 1024)


def test_the_pinned_budget_is_a_fraction_of_host_memory() -> None:
    assert pinned_budget_bytes(64 * GB) == int(64 * GB * 0.05)
    assert pinned_budget_bytes(0) == 0
    assert pinned_budget_bytes(64 * GB, fraction=0.0) == 0


def test_a_plan_on_a_known_link_pins_and_overlaps() -> None:
    plan = plan_staging(GB, 25.0, host_bytes=64 * GB)
    assert plan.pinned
    assert plan.overlapped
    assert plan.chunk_bytes == chunk_bytes_for_link(25.0)


def test_an_unknown_link_yields_the_behavior_a_caller_already_had() -> None:
    plan = plan_staging(GB, 0.0)
    assert plan.chunk_bytes == MIN_CHUNK_BYTES
    assert plan.depth == 1
    assert not plan.overlapped


def test_a_tight_budget_shrinks_the_depth_and_not_the_chunk() -> None:
    chunk = chunk_bytes_for_link(25.0)
    plan = plan_staging(GB, 25.0, prepare_us=1e6, host_bytes=int(chunk * 3 / 0.05))
    assert plan.chunk_bytes == chunk
    assert plan.depth == 3
    assert plan.pinned


def test_a_budget_too_small_for_one_chunk_gives_the_pages_back() -> None:
    plan = plan_staging(GB, 25.0, host_bytes=1024)
    assert not plan.pinned
    assert plan.chunk_bytes == chunk_bytes_for_link(25.0)


def test_the_budget_is_shared_across_the_streams_feeding_each_device() -> None:
    chunk = chunk_bytes_for_link(25.0)
    plan = plan_staging(GB, 25.0, streams=4, prepare_us=1e6, host_bytes=int(chunk * 8 / 0.05))
    assert plan.streams == 4
    assert plan.depth == 2  # eight chunks of budget over four rings
    assert plan.buffer_bytes == chunk * 2 * 4


def test_a_small_transfer_is_not_pinned_however_much_memory_there_is() -> None:
    assert not plan_staging(1024, 25.0, host_bytes=1024 * GB).pinned


def test_effective_bandwidth_derates_a_plan_that_cannot_overlap() -> None:
    pinned_deep = StagingPlan(chunk_bytes=1 << 20, depth=4, pinned=True)
    assert effective_gbps(pinned_deep, 25.0) == 25.0
    assert effective_gbps(StagingPlan(depth=1, pinned=True), 25.0) == 12.5


def test_effective_bandwidth_derates_a_pageable_plan_for_the_bounce_buffer() -> None:
    assert effective_gbps(StagingPlan(depth=4, pinned=False), 25.0) == 12.5
    assert effective_gbps(StagingPlan(depth=1, pinned=False), 25.0) == 6.25


def test_an_unknown_link_has_no_effective_rate() -> None:
    assert effective_gbps(StagingPlan(depth=4, pinned=True), 0.0) == 0.0


def test_duration_is_bytes_over_the_rate_the_plan_actually_achieves() -> None:
    plan = StagingPlan(chunk_bytes=1 << 20, depth=4, pinned=True)
    assert staging_seconds(25_000_000_000, plan, 25.0) == pytest.approx(1.0)
    assert staging_seconds(0, plan, 25.0) == 0.0
    assert staging_seconds(1000, plan, 0.0) == 0.0


def test_concurrent_streams_share_the_host_link_rather_than_multiplying_it() -> None:
    """The optimistic reading would have a caller size a stage for bandwidth that is not there."""
    one = StagingPlan(chunk_bytes=1 << 20, depth=4, pinned=True, streams=1)
    four = StagingPlan(chunk_bytes=1 << 20, depth=4, pinned=True, streams=4)
    assert staging_seconds(GB, one, 25.0) == staging_seconds(GB, four, 25.0)


def test_the_plan_summarizes_flat() -> None:
    plan = StagingPlan(chunk_bytes=1 << 20, depth=3, pinned=True, streams=2)
    assert plan.summary() == {
        "chunk_bytes": 1 << 20,
        "depth": 3,
        "pinned": True,
        "streams": 2,
        "buffer_bytes": (1 << 20) * 6,
    }
