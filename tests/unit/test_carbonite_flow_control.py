"""Carbonite credit-window flow control: the authority over the shuffle window.

One credit = one in-flight RecordBatch slot, so the granted window bounds a shuffle
channel's buffered memory. `ResourceManager.grant_credits` replaces the engine's
hardcoded `DEFAULT_CREDITS`; these tests pin the clamp band it derives from config.
"""

from __future__ import annotations

import pytest

from batcher.carbonite import ResourceManager
from batcher.carbonite.policies import AIMDFlowControl, credit_ceiling
from batcher.config import Config, FlowControlConfig, config_context

pytestmark = pytest.mark.unit


def _aimd(**fc):
    return AIMDFlowControl(Config().replace(flow_control=FlowControlConfig(**fc)))


def test_aimd_starts_at_default_window():
    assert _aimd(default_credits=4).window == 4


def test_credit_ceiling_shrinks_for_wide_learned_rows():
    # C9: the credit→bytes conversion assumes a `morsel_bytes` batch. When the learned
    # row width fills a `morsel_rows` batch to far more than that (embeddings/blobs), the
    # ceiling must hand out fewer credits so the channel's buffered bytes stay in budget.
    from batcher.carbonite.base import ResourceContext
    from batcher.carbonite.memory.learned import LearnedMemoryModel
    from batcher.carbonite.policies import (
        StaticCreditFlowControl,
        credit_ceiling,
        learned_channel_morsel_bytes,
    )

    cfg = Config().replace(
        flow_control=FlowControlConfig(
            default_credits=64, credit_ceiling_factor=8, credit_byte_budget=64 << 20
        )
    )
    narrow = credit_ceiling(cfg)  # byte bound at the assumed morsel_bytes
    wide_model = LearnedMemoryModel(
        _bytes_per_row={"aggregate": 2000.0},  # 2 KB/row → a morsel is ~32 MB, not 1 MB
        _alpha=0.5,
        _clamp=4.0,
        _row_bytes=8,
        _spill_per_row={},
    )
    ctx = ResourceContext(config=cfg, memory_model=wide_model)
    eff = learned_channel_morsel_bytes(ctx)
    assert eff is not None and eff > cfg.execution.morsel_bytes
    assert credit_ceiling(cfg, eff) < narrow
    # The static policy grants the tighter window for the wide-row channel.
    wide_grant = StaticCreditFlowControl().grant(1000, ctx)
    narrow_grant = StaticCreditFlowControl().grant(1000, ResourceContext(config=cfg))
    assert wide_grant < narrow_grant


def test_aimd_slow_starts_then_recovers_along_the_cubic_curve():
    # Before the first congestion the window is in slow-start and DOUBLES each headroom
    # round (TCP slow-start) so it fills the bandwidth-delay product in log2 rounds; the
    # first congestion exits slow-start into congestion avoidance.
    #
    # Congestion avoidance recovers along CUBIC, not `+alpha`. The window congestion was
    # found at (16) is a *measured* capacity, so climbing back to it one credit per round
    # would spend most of the transfer below a value the channel already proved safe — on a
    # cross-node shuffle every one of those rounds is a network round trip. The cubic is
    # steepest far below `w_max` and flattens as it approaches, so it recovers in a couple of
    # rounds and then sits near the known-good window.
    a = _aimd(default_credits=4, aimd_alpha=1)
    assert a.observe(congested=False) == 8  # slow-start: 4 -> 8
    assert a.observe(congested=False) == 16  # slow-start: 8 -> 16
    assert a.observe(congested=True) == 8  # congestion: cut, and leave slow-start
    first = a.observe(congested=False)
    second = a.observe(congested=False)
    assert first > 9, "additive increase would have crawled to 9"
    assert first < 16, "recovery must approach the known-good window from below"
    assert first < second <= 16


def test_cubic_recovery_is_never_slower_than_additive_increase():
    # CUBIC's TCP-friendly region: growth is the larger of the cubic and the additive law, so
    # the change can only ever make recovery faster, never slower — including on a small
    # window where the cubic is flat and `+alpha` is what actually moves it.
    cfg = Config().replace(flow_control=FlowControlConfig(default_credits=4, aimd_alpha=1))
    ceiling = credit_ceiling(cfg)
    for beta in (0.5, 0.9):
        cubic = _aimd(default_credits=4, aimd_alpha=1, aimd_beta=beta)
        for _ in range(3):
            cubic.observe(congested=False)
        cubic.observe(congested=True)
        after_backoff = cubic.window
        windows = [cubic.observe(congested=False) for _ in range(6)]
        # Both laws are clamped by the same memory-safe ceiling, so the comparison is
        # against additive increase *under that ceiling*.
        additive = [min(after_backoff + i, ceiling) for i in range(1, 7)]
        assert all(c >= a for c, a in zip(windows, additive, strict=True)), (
            f"beta={beta}: {windows} fell below additive increase {additive}"
        )


def test_cubic_growth_never_passes_the_memory_safe_ceiling():
    a = _aimd(default_credits=4, aimd_alpha=1)
    for _ in range(6):
        a.observe(congested=False)
    a.observe(congested=True)
    ceiling = a.window
    for _ in range(200):
        ceiling = max(ceiling, a.observe(congested=False))
    cfg = Config().replace(flow_control=FlowControlConfig(default_credits=4, aimd_alpha=1))
    assert ceiling <= credit_ceiling(cfg)


def test_aimd_warm_started_channel_skips_slow_start():
    # A recurring shuffle warm-starts at a learned window that already reflects prior
    # congestion; it must NOT slow-start (exponential ramp would overshoot that value).
    a = AIMDFlowControl(
        Config().replace(flow_control=FlowControlConfig(default_credits=4, aimd_alpha=1)),
        initial_window=10,
    )
    assert a.observe(congested=False) == 11  # additive from the warm start, not 20


def test_aimd_shrinks_multiplicatively_on_congestion():
    a = _aimd(default_credits=16, aimd_beta=0.5)
    assert a.observe(congested=True) == 8
    assert a.observe(congested=True) == 4


def test_aimd_stays_within_band():
    a = _aimd(default_credits=4, credit_ceiling_factor=2, aimd_alpha=1, aimd_beta=0.5)
    for _ in range(50):  # relentless growth clamps at the ceiling (4 * 2 = 8)
        a.observe(congested=False)
    assert a.window == 8
    for _ in range(50):  # relentless congestion clamps at the floor (1)
        a.observe(congested=True)
    assert a.window == 1


def test_aimd_grant_ignores_request_and_returns_window():
    a = _aimd(default_credits=4)
    a.observe(congested=False)  # slow-start: window -> 8
    rm = ResourceManager(flow_control=a)
    assert rm.grant_credits(999) == 8  # AIMD owns the window, not the request


def test_unset_request_falls_back_to_default_window():
    # An operator with no `c_max_credits` estimate (request <= 0) gets the config
    # default window rather than stalling the channel at zero credits.
    assert ResourceManager().grant_credits(0) == FlowControlConfig().default_credits
    assert ResourceManager().grant_credits(-7) == FlowControlConfig().default_credits


def test_reasonable_request_passes_through():
    assert ResourceManager().grant_credits(8) == 8


def test_oversized_request_is_clamped_to_ceiling():
    fc = FlowControlConfig()
    ceiling = fc.default_credits * fc.credit_ceiling_factor
    assert ResourceManager().grant_credits(10_000) == ceiling


def test_window_is_config_driven():
    # The window tracks config, not a hardcoded constant — the single source of truth.
    # Explicit factor + a generous byte budget so the count ceiling (not the byte cap)
    # is what this asserts, independent of the shipped defaults.
    cfg = Config().replace(
        flow_control=FlowControlConfig(
            default_credits=12, credit_ceiling_factor=8, credit_byte_budget=1 << 40
        )
    )
    with config_context(cfg):
        assert ResourceManager().grant_credits(0) == 12
        assert ResourceManager().grant_credits(10_000) == 12 * 8


def test_grant_never_returns_zero():
    cfg = Config().replace(flow_control=FlowControlConfig(default_credits=1))
    with config_context(cfg):
        assert ResourceManager().grant_credits(0) >= 1
        assert ResourceManager().grant_credits(-1) >= 1
