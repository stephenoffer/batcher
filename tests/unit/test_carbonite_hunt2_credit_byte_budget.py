"""Wide-row credit-byte-budget (C53) must be honored on EVERY credit-granting path.

The static `StaticCreditFlowControl.grant` corrects its ceiling for a learned wide-row
width so a channel's buffered bytes stay within `credit_byte_budget`. The learned-window
branch of `grant_credits` and the AIMD `adaptive_flow_control` path must apply the same
correction — otherwise a wide-row (embeddings/blobs) shuffle grows its window to the
un-corrected *count* ceiling and a fast producer buffers far past the byte budget (a
credit over-issue → OOM). This pins that the three paths agree.
"""

from __future__ import annotations

import dataclasses

from batcher.carbonite import ResourceManager
from batcher.carbonite.memory.learned import LearnedMemoryModel
from batcher.carbonite.policies import (
    StaticCreditFlowControl,
    credit_ceiling,
    learned_channel_morsel_bytes,
    record_shuffle_window,
)
from batcher.config import active_config
from batcher.metadata import MetadataHub
from batcher.metadata.backends import InProcessBackend


def _manager_with_wide_model() -> ResourceManager:
    hub = MetadataHub(InProcessBackend())
    record_shuffle_window(hub, "s", 64)  # a large learned window (the count ceiling)
    rm = ResourceManager(hub=hub)
    # A wide-row learned model: ~200 KB/row (blobs/embeddings) fills a morsel far past the
    # assumed `morsel_bytes`, so the byte bound must shrink the credit window.
    model = LearnedMemoryModel(
        _bytes_per_row={"hash_join": 200_000.0},
        _alpha=0.5,
        _clamp=4.0,
        _row_bytes=8,
        _spill_per_row={},
    )
    rm._mem_model = model
    rm._ctx = dataclasses.replace(rm._ctx, memory_model=model)
    return rm


def test_learned_window_grant_honors_byte_budget() -> None:
    rm = _manager_with_wide_model()
    byte_ceiling = credit_ceiling(active_config(), learned_channel_morsel_bytes(rm._ctx))
    # Without the fix this returned the un-corrected count ceiling (64), a 64x over-issue.
    assert rm.grant_credits(0, signature="s") <= byte_ceiling
    # It must match what the static path grants for the same wide-row channel.
    assert rm.grant_credits(0, signature="s") == StaticCreditFlowControl().grant(0, rm._ctx)


def test_adaptive_controller_ceiling_honors_byte_budget() -> None:
    rm = _manager_with_wide_model()
    byte_ceiling = credit_ceiling(active_config(), learned_channel_morsel_bytes(rm._ctx))
    ctrl = rm.adaptive_flow_control(signature="s")
    # AIMD may never grow past the byte-corrected ceiling, even on many headroom rounds.
    for _ in range(50):
        ctrl.observe(congested=False)
    assert ctrl.window <= byte_ceiling


def test_the_channel_byte_budget_is_held_under_a_share_of_real_memory(monkeypatch):
    """The buffered-bytes budget must know how much memory the machine has.

    `credit_byte_budget` is a fixed 256 MiB per channel and `shuffle_fetch_fan_in` channels
    fetch at once — 8 GiB in flight on the defaults. That is noise on a 512 GiB node and half
    the RAM of a 16 GiB one. Admission was already memory-aware; backpressure was not, so a
    node could be admitted for a query and then OOM'd by the transit buffers carrying it.
    """
    from batcher.carbonite.policies import flow_control as policies
    from batcher.config import active_config

    cfg = active_config()
    gib = 1 << 30
    fan_in = cfg.flow_control.shuffle_fetch_fan_in

    def budget_on(total_bytes):
        monkeypatch.setattr(policies, "total_memory_bytes", lambda: total_bytes)
        return policies._channel_byte_budget(cfg)

    # A fat node keeps the configured budget — this caps, it never throttles a machine that
    # can afford the buffer.
    assert budget_on(512 * gib) == cfg.flow_control.credit_byte_budget
    # A small node is held down, and the whole in-flight footprint stays a modest share of RAM.
    small = budget_on(16 * gib)
    assert small < cfg.flow_control.credit_byte_budget
    assert small * fan_in <= 16 * gib * 0.15
    # Monotone in machine size, and never below one morsel (a zero would deadlock the window).
    assert budget_on(2 * gib) < small
    assert budget_on(64 * (1 << 20)) >= cfg.execution.morsel_bytes
    # An unreadable total falls back to the configured value rather than to zero.
    assert budget_on(0) == cfg.flow_control.credit_byte_budget


def test_a_small_node_lowers_the_credit_ceiling_itself(monkeypatch):
    """The headroom cap has to reach the ceiling callers actually use, not just the helper."""
    from batcher.carbonite.policies import flow_control as policies
    from batcher.config import active_config

    cfg = active_config()
    monkeypatch.setattr(policies, "total_memory_bytes", lambda: 512 << 30)
    roomy = policies.credit_ceiling(cfg)
    monkeypatch.setattr(policies, "total_memory_bytes", lambda: 2 << 30)
    cramped = policies.credit_ceiling(cfg)
    assert 1 <= cramped < roomy
