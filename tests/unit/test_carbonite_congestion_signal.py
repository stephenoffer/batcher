"""The credit window's control loop, closed on the channel's own evidence.

Before the starvation counter existed, `AIMDFlowControl` had exactly one input: whether the
*node* was past its spill threshold. On a healthy node that never fires, so every round read
"grow" and the window climbed to its ceiling whether or not the extra credits moved a byte.
These tests pin the third state that fixes it — a saturated channel *holds* — and the
measurement plumbing that makes the verdict trustworthy round to round.
"""

from __future__ import annotations

import pytest

from batcher.carbonite.policies import (
    AIMDFlowControl,
    ChannelCongestion,
    CongestionSignal,
    StarvationMeter,
    occupancy_from_starvation,
)
from batcher.config import Config, FlowControlConfig


def _cfg(**kw: object) -> Config:
    return Config(flow_control=FlowControlConfig(**kw))  # type: ignore[arg-type]


# --- StarvationMeter: cumulative clocks into a per-round ratio ----------------------


def test_the_first_sample_has_no_interval_to_report() -> None:
    """A meter with no baseline has measured nothing, which is not "saturated"."""
    assert StarvationMeter().sample((0.0, 0.0)) is None


def test_a_round_is_the_difference_of_the_totals_not_their_ratio() -> None:
    """The whole reason the transport hands back totals rather than a ratio.

    The first second is fully starved and the next nine are not. The lifetime ratio at the
    end is 0.1 and falling; the *round* is 0.0, which is what the controller must act on.
    """
    meter = StarvationMeter()
    meter.sample((0.0, 0.0))
    assert meter.sample((1.0, 1.0)) == 1.0
    assert meter.sample((1.0, 10.0)) == 0.0


def test_a_long_shuffle_does_not_wash_out_a_starved_round() -> None:
    """The failure mode a lifetime ratio has and an interval does not.

    After a thousand saturated seconds, one fully starved second moves the lifetime ratio to
    0.001 — indistinguishable from saturated — while the interval reads it at 1.0.
    """
    meter = StarvationMeter()
    meter.sample((0.0, 1000.0))
    assert meter.sample((1.0, 1001.0)) == 1.0


def test_a_counter_reset_re_baselines_instead_of_reporting_a_negative_round() -> None:
    """`reset_peer_transfers` rewinds the totals underneath a live meter.

    Subtracting the larger baseline would give a negative interval, which clamps to zero and
    reads as a perfectly saturated round — telling the controller to stop growing on the
    strength of a bookkeeping event.
    """
    meter = StarvationMeter()
    meter.sample((5.0, 20.0))
    assert meter.sample((0.0, 0.0)) is None, "a reset is not a measurement"
    assert meter.sample((1.0, 2.0)) == 0.5, "and the next round measures from the new base"


def test_too_little_transfer_to_divide_reports_no_opinion() -> None:
    """A round served entirely from locality touched no wire and has nothing to say."""
    meter = StarvationMeter()
    meter.sample((0.0, 0.0))
    assert meter.sample((0.0, 1e-9)) is None


def test_two_meters_share_one_node_level_measurement() -> None:
    """The transport's clocks are process-wide, and the controllers read them, not each other.

    Deliberate rather than a shortcut: a window is granted per channel but the memory it buys
    is the node's, so "was this worker ever waiting on the wire" is the question that should
    size it. The consequence worth pinning is that each meter reports the interval since *its
    own* last sample over *all* channels — so two controllers sampling at different times
    legitimately see different intervals of the same shared stream.
    """
    a, b = StarvationMeter(), StarvationMeter()
    a.sample((0.0, 0.0))
    b.sample((0.0, 0.0))
    # `a` samples mid-flight; `b` waits and sees the whole span, including what `a` saw.
    assert a.sample((1.0, 1.0)) == 1.0
    assert b.sample((1.0, 3.0)) == pytest.approx(1 / 3)


def test_occupancy_is_the_complement_of_starvation() -> None:
    """The transport measures waiting; the config thresholds are stated as fullness."""
    assert occupancy_from_starvation(0.25) == 0.75
    assert occupancy_from_starvation(None) is None
    assert occupancy_from_starvation(2.0) == 0.0, "clamped, never negative"


# --- ChannelCongestion: fusing the two facts ---------------------------------------


def test_memory_pressure_outranks_a_starving_channel() -> None:
    """A slow shuffle is recoverable and an OOM-killed worker is not."""
    fused = ChannelCongestion(_cfg())
    assert fused.observe(pressured=True, occupancy=0.0) is CongestionSignal.CONGESTED


def test_an_unmeasured_channel_is_permissive_not_saturated() -> None:
    """A window that has never been tested has not earned the right to stop growing.

    This is also what keeps a build with no starvation counter behaving exactly as the engine
    did before the measurement existed.
    """
    fused = ChannelCongestion(_cfg())
    assert fused.observe(pressured=False, occupancy=None) is CongestionSignal.STARVED


def test_the_thresholds_are_a_band_and_the_band_holds_the_last_verdict() -> None:
    """The Schmitt trigger `backpressure_high`/`backpressure_low` were always documented as.

    A single threshold on a noisy ratio flaps, and a flapping verdict drives a window that
    grows and cuts on alternate rounds — worse than either decision taken consistently.
    """
    fused = ChannelCongestion(_cfg(backpressure_low=0.4, backpressure_high=0.7))
    assert fused.observe(pressured=False, occupancy=0.9) is CongestionSignal.SATURATED
    # Inside the band: the previous verdict stands rather than being re-decided.
    assert fused.observe(pressured=False, occupancy=0.55) is CongestionSignal.SATURATED
    assert fused.observe(pressured=False, occupancy=0.2) is CongestionSignal.STARVED
    assert fused.observe(pressured=False, occupancy=0.55) is CongestionSignal.STARVED


def test_the_retained_verdict_is_readable_without_taking_a_sample() -> None:
    """`stats()` reports it, and reading a diagnosis must not perturb the controller."""
    fused = ChannelCongestion(_cfg())
    fused.observe(pressured=False, occupancy=1.0)
    assert fused.signal is CongestionSignal.SATURATED
    assert fused.signal is CongestionSignal.SATURATED


# --- AIMDFlowControl: the third state ----------------------------------------------


def test_a_saturated_channel_holds_instead_of_climbing_to_the_ceiling() -> None:
    """The headline regression this closes.

    With memory as the only signal, a healthy channel grew every round until it hit the
    ceiling — buying up to a full `credit_byte_budget` of buffered memory per channel for no
    throughput at all, and manufacturing the very pressure it would then back off on.
    """
    ctrl = AIMDFlowControl(_cfg())
    start = ctrl.window
    for _ in range(50):
        ctrl.observe_signal(CongestionSignal.SATURATED)
    assert ctrl.window == start, "a channel that never waits needs no more credits"
    assert ctrl.window < ctrl.stats()["ceiling"], "and must not have reached the ceiling"


def test_a_starved_channel_still_grows() -> None:
    """Holding must not cost the throughput the window exists to provide."""
    ctrl = AIMDFlowControl(_cfg())
    start = ctrl.window
    ctrl.observe_signal(CongestionSignal.STARVED)
    assert ctrl.window > start


def test_congestion_still_cuts_multiplicatively() -> None:
    ctrl = AIMDFlowControl(_cfg(default_credits=16, aimd_beta=0.5))
    for _ in range(4):
        ctrl.observe_signal(CongestionSignal.STARVED)
    before = ctrl.window
    ctrl.observe_signal(CongestionSignal.CONGESTED)
    assert ctrl.window < before


def test_holding_does_not_advance_the_recovery_clock() -> None:
    """Otherwise a converged window jumps to the ceiling the moment it starves once.

    CUBIC's growth is `w_max + C(t - K)**3`, so `t` must count rounds spent *recovering*. If
    held rounds advanced it, a channel that sat saturated for a minute and then starved once
    would evaluate the curve far out and leap straight to its ceiling — the stale-window
    behaviour in a new costume.
    """
    ctrl = AIMDFlowControl(_cfg(default_credits=16))
    ctrl.observe_signal(CongestionSignal.CONGESTED)  # sets w_max and starts the clock
    after_backoff = ctrl.window
    for _ in range(200):
        ctrl.observe_signal(CongestionSignal.SATURATED)
    assert ctrl.window == after_backoff
    ctrl.observe_signal(CongestionSignal.STARVED)
    assert ctrl.window < ctrl.stats()["ceiling"], "one starved round is not a full recovery"


def test_the_boolean_shorthand_keeps_its_historical_meaning() -> None:
    """`observe(congested=)` is the two-state view, and must behave exactly as it always did."""
    grew = AIMDFlowControl(_cfg())
    start = grew.window
    grew.observe(congested=False)
    assert grew.window > start, "an uncongested round still grows on the two-state law"

    cut = AIMDFlowControl(_cfg())
    for _ in range(4):
        cut.observe(congested=False)
    before = cut.window
    cut.observe(congested=True)
    assert cut.window < before


def test_stats_report_convergence_as_well_as_backoff() -> None:
    """A window pinned at its ceiling reading STARVED is throttled by the ceiling; the same
    window reading SATURATED found its bandwidth-delay product. Only `hold_rate` tells them
    apart."""
    ctrl = AIMDFlowControl(_cfg())
    for _ in range(3):
        ctrl.observe_signal(CongestionSignal.STARVED)
    for _ in range(7):
        ctrl.observe_signal(CongestionSignal.SATURATED)
    stats = ctrl.stats()
    assert stats["rounds"] == 10, "a held round is still a round"
    assert stats["holds"] == 7
    assert stats["hold_rate"] == 0.7


def test_a_regrant_does_not_lose_the_lifetime_counters() -> None:
    """`rewindow` restarts the window for a new query; the counters describe the channel."""
    ctrl = AIMDFlowControl(_cfg())
    ctrl.observe_signal(CongestionSignal.SATURATED)
    ctrl.rewindow(8)
    assert ctrl.stats()["holds"] == 1


def test_an_unmeasurable_round_holds_the_last_verdict_instead_of_growing() -> None:
    """ "No opinion" is not "starving", and conflating them ratchets the window open.

    `StarvationMeter.sample` returns `None` for any round that carried too little transfer to
    divide — one served from locality, or one that moved a single tiny bucket. Treating that
    as `STARVED` tells the controller to grow whatever the wire said, so a shuffle alternating
    between measurable and unmeasurable rounds climbs to its ceiling regardless. Only a
    controller that has *never* measured is permissive, and that is its initial state.
    """
    fused = ChannelCongestion(_cfg())
    assert fused.observe(pressured=False, occupancy=1.0) is CongestionSignal.SATURATED
    assert fused.observe(pressured=False, occupancy=None) is CongestionSignal.SATURATED
    assert fused.signal is CongestionSignal.SATURATED
