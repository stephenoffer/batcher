"""Carbonite credit-window flow control: the authority over the shuffle window.

One credit = one in-flight RecordBatch slot, so the granted window bounds a shuffle
channel's buffered memory. `ResourceManager.grant_credits` replaces the engine's
hardcoded `DEFAULT_CREDITS`; these tests pin the clamp band it derives from config.
"""

from __future__ import annotations

import pytest

from batcher.carbonite import ResourceManager
from batcher.carbonite.policies import AIMDFlowControl
from batcher.config import Config, FlowControlConfig, config_context

pytestmark = pytest.mark.unit


def _aimd(**fc):
    return AIMDFlowControl(Config().replace(flow_control=FlowControlConfig(**fc)))


def test_aimd_starts_at_default_window():
    assert _aimd(default_credits=4).window == 4


def test_aimd_grows_additively_when_uncongested():
    a = _aimd(default_credits=4, aimd_alpha=1)
    assert a.observe(congested=False) == 5
    assert a.observe(congested=False) == 6


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
    a.observe(congested=False)  # window -> 5
    rm = ResourceManager(flow_control=a)
    assert rm.grant_credits(999) == 5  # AIMD owns the window, not the request


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
    cfg = Config().replace(flow_control=FlowControlConfig(default_credits=16))
    with config_context(cfg):
        assert ResourceManager().grant_credits(0) == 16
        assert ResourceManager().grant_credits(10_000) == 16 * 16


def test_grant_never_returns_zero():
    cfg = Config().replace(flow_control=FlowControlConfig(default_credits=1))
    with config_context(cfg):
        assert ResourceManager().grant_credits(0) >= 1
        assert ResourceManager().grant_credits(-1) >= 1
