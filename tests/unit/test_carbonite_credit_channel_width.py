"""The credit budget has to be divided by the channels that exist, not the ones allowed.

`credit_byte_budget` bounds a shuffle channel's buffered memory, and the whole-shuffle
share is that budget divided across the channels fetching at once. Dividing by
`shuffle_fetch_fan_in` instead divides by a *cap on* concurrency rather than a measurement
of it, and the two differ in both directions: a reducer with three upstreams gets a share
sized for thirty-two and buffers less than it safely could, while one with a hundred and
twenty-eight gets thirty-two channels' worth each and buffers far more than the node has.

`credit_ceiling` grew a `channels` parameter for exactly this, with a docstring describing
the fix — and no production caller ever passed it. The correction existed, was unit-tested
in isolation, and was inert in every real run, which is the failure mode worth a test file:
not a wrong answer, but a right one nothing asks for.

Measured on a 16 GiB node (a 1,638 MiB transit share) with 128 channels open: the blind
ceiling permits 6,528 MiB in flight, 4x the share. Divided by the real width it is
1,536 MiB, inside it.
"""

from __future__ import annotations

import pytest

from batcher.carbonite import ResourceManager
from batcher.carbonite.base import ResourceContext
from batcher.carbonite.policies import flow_control as fc_mod
from batcher.carbonite.policies.flow_control import (
    AIMDFlowControl,
    StaticCreditFlowControl,
    credit_ceiling,
)
from batcher.config import Config, ExecutionConfig

pytestmark = pytest.mark.unit

#: A node small enough that the memory-share term binds rather than the configured cap.
#: On a large box `min(configured, headroom)` is always `configured` and the division by
#: channel count is invisible — a test written on the host's real memory asserts nothing.
_SMALL_NODE = 16 << 30
_TRANSIT_SHARE = int(_SMALL_NODE * 0.10)


@pytest.fixture
def small_node(monkeypatch):
    """Pin the machine's memory so the headroom term, not the configured cap, decides."""
    monkeypatch.setattr(fc_mod, "total_memory_bytes", lambda: _SMALL_NODE)
    return Config().replace(execution=ExecutionConfig(morsel_bytes=1 << 20))


def _in_flight_bytes(credits: int, channels: int, morsel_bytes: int = 1 << 20) -> int:
    return credits * channels * morsel_bytes


# --- the ceiling itself -------------------------------------------------------


def test_a_wide_shuffle_stays_inside_the_transit_share(small_node) -> None:
    """The OOM this prevents: every channel granted a share sized for far fewer."""
    channels = 128
    aware = credit_ceiling(small_node, channels=channels)
    assert _in_flight_bytes(aware, channels) <= _TRANSIT_SHARE


def test_the_blind_ceiling_would_have_exceeded_it(small_node) -> None:
    """The negative control, so the test above is known to be measuring something."""
    channels = 128
    blind = credit_ceiling(small_node)  # divides by the configured fan-in cap
    assert _in_flight_bytes(blind, channels) > _TRANSIT_SHARE


def test_a_narrow_shuffle_is_not_starved(small_node) -> None:
    """The correction runs both ways: three channels should not be sized for thirty-two."""
    narrow = credit_ceiling(small_node, channels=3)
    blind = credit_ceiling(small_node)
    assert narrow >= blind


def test_more_channels_never_raises_the_per_channel_window(small_node) -> None:
    """Monotone in the divisor, which is what makes the bound a bound."""
    ceilings = [credit_ceiling(small_node, channels=n) for n in (2, 8, 32, 128, 512)]
    assert ceilings == sorted(ceilings, reverse=True)


def test_the_window_never_falls_below_one(small_node) -> None:
    """A zero-credit channel stalls forever; the bound must never produce one."""
    assert credit_ceiling(small_node, channels=100_000) >= 1


def test_an_unknown_width_keeps_the_configured_behaviour(small_node) -> None:
    """A caller that cannot measure its width must get exactly what it got before."""
    assert credit_ceiling(small_node, channels=None) == credit_ceiling(small_node)
    assert credit_ceiling(small_node, channels=0) == credit_ceiling(small_node)


# --- the wiring, which is the part that was missing ---------------------------


def test_the_static_policy_reads_the_measured_width(small_node) -> None:
    """`StaticCreditFlowControl` is the default policy and must honour the context."""
    wide = ResourceContext(config=small_node, shuffle_channels=128)
    blind = ResourceContext(config=small_node)
    policy = StaticCreditFlowControl()
    assert policy.grant(1000, wide) < policy.grant(1000, blind)


def test_the_adaptive_policy_reads_it_too(small_node) -> None:
    """AIMD grows *toward* the ceiling, so a wrong ceiling is the value it converges to.

    A fix applied only to the static path would leave the adaptive path — the one that
    deliberately climbs until it finds the bound — buffering past it.
    """
    wide = AIMDFlowControl(small_node, channels=128)
    blind = AIMDFlowControl(small_node)
    ctx = ResourceContext(config=small_node)
    for _ in range(40):  # climb hard, with no congestion to hold either back
        wide.observe(congested=False)
        blind.observe(congested=False)
    assert wide.grant(0, ctx) < blind.grant(0, ctx)
    assert _in_flight_bytes(wide.grant(0, ctx), 128) <= _TRANSIT_SHARE


def test_the_manager_threads_the_width_to_the_policy(small_node) -> None:
    """The end of the chain: `grant_credits(channels=)` must reach the ceiling."""
    from batcher.config import config_context

    with config_context(small_node):
        rm = ResourceManager()
        assert rm.grant_credits(1000, channels=128) < rm.grant_credits(1000)


def test_the_manager_default_is_unchanged(small_node) -> None:
    """Omitting the width must be byte-for-byte the previous behaviour."""
    from batcher.config import config_context

    with config_context(small_node):
        rm = ResourceManager()
        assert rm.grant_credits(1000) == credit_ceiling(small_node)
        assert rm.grant_credits(1000, channels=0) == credit_ceiling(small_node)


def test_a_grant_is_still_at_least_one_credit(small_node) -> None:
    from batcher.config import config_context

    with config_context(small_node):
        rm = ResourceManager()
        assert rm.grant_credits(0, channels=100_000) >= 1
