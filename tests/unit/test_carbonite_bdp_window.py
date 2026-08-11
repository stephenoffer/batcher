"""Sizing a credit window from the path, instead of probing for it.

AIMD and CUBIC probe because a TCP sender cannot measure what it needs. Batcher can: the
transport keeps a min-filter on round-trip time and a max-filter on delivery rate, so the
bandwidth-delay product is a measurement rather than something to be discovered by overshooting.

These tests pin the two results that follow. The window's factor of two is *forced* by the
refill batching rather than chosen, and an even split of a reducer's credit budget is worse
than a size-proportional one by exactly the shuffle's skew factor.
"""

from __future__ import annotations

import pytest

from batcher.carbonite.policies import (
    REFILL_WINDOW_GAIN,
    AIMDFlowControl,
    bdp_window,
    proportional_windows,
)
from batcher.config import Config, FlowControlConfig

# --- the window is the pipe, doubled ------------------------------------------------


def test_the_window_is_the_bandwidth_delay_product_in_credits() -> None:
    """An 8 MiB pipe with 1 MiB morsels holds 8 batches; the window carries twice that."""
    assert bdp_window(8 << 20, 1 << 20) == 8 * REFILL_WINDOW_GAIN


def test_the_gain_is_what_the_refill_watermark_forces() -> None:
    """Not a tuning constant — arithmetic.

    The consumer returns credits in bulk once half the window has drained, so on top of the
    `L x R` batches in flight sit up to `w / 2` consumed-but-unacknowledged ones. The producer
    stalls unless `w >= L x R + w / 2`, which rearranges to `w >= 2 x BDP`. A window set to
    exactly the product idles the producer for about half of every round trip.
    """
    bdp_batches = 9
    window = bdp_window(bdp_batches * (1 << 20), 1 << 20)
    in_flight = bdp_batches
    unacknowledged_allowance = window // 2
    assert window >= in_flight + unacknowledged_allowance
    # And it is not needlessly larger than that: one more credit than the bound requires
    # would be memory bought for nothing.
    assert window - 1 < in_flight + (window - 1) / 2 or window == 2 * bdp_batches


def test_a_partial_batch_still_needs_a_whole_credit() -> None:
    """Ceiling, not rounding: a window a batch short of the product idles every round trip."""
    assert bdp_window((1 << 20) + 1, 1 << 20) == 2 * REFILL_WINDOW_GAIN


def test_no_measurement_is_not_a_pipe_of_zero_width() -> None:
    """A controller handed zero would pin every window to its floor forever."""
    assert bdp_window(0, 1 << 20) == 1
    assert bdp_window(-5, 1 << 20) == 1
    assert bdp_window(1 << 20, 0) == 1


def test_an_informed_start_skips_the_search_it_would_have_run() -> None:
    """Slow start is a search, and a search is only worth running when nothing is known.

    Doubling from 16 credits to 64 costs two round trips, which on a 100 ms link is 200 ms a
    short bucket never gets back — it finishes below the window it needed. A measured product
    is the answer that search converges to.
    """
    cfg = Config(flow_control=FlowControlConfig(default_credits=16))
    blind = AIMDFlowControl(cfg)
    informed = AIMDFlowControl(cfg, bdp_window=40)
    assert blind.window == 16 and blind.stats()["slow_start"] == 1
    assert informed.window == 40 and informed.stats()["slow_start"] == 0


def test_a_product_below_the_default_cannot_shrink_the_start() -> None:
    """The one way this measurement could do harm, closed.

    A loopback path has almost no propagation delay, so its product really is a credit or
    two — correct for that path, and a disastrous opening window for the cross-node fetch the
    same process makes next. An informed start also switches slow start off, so there would be
    no exponential ramp to rescue it. The product may only ever lift the configured floor.
    """
    cfg = Config(flow_control=FlowControlConfig(default_credits=16))
    loopback = AIMDFlowControl(cfg, bdp_window=2)
    assert loopback.window == 16, "a short path must not set the window for a long one"
    assert loopback.stats()["slow_start"] == 1, (
        "a product below the default informed nothing, so the search still has work to do"
    )


def test_a_learned_window_outranks_a_measured_product() -> None:
    """This shuffle's own history beats the process's paths, which it may not be typical of."""
    cfg = Config()
    ctrl = AIMDFlowControl(cfg, initial_window=7, bdp_window=40)
    assert ctrl.window == 7


def test_the_informed_start_is_still_held_under_the_ceiling() -> None:
    """A measurement moves the starting point, never the memory envelope."""
    cfg = Config()
    ctrl = AIMDFlowControl(cfg, bdp_window=10_000_000)
    assert ctrl.window == ctrl.stats()["ceiling"]


def test_an_unmeasured_process_keeps_the_probing_ramp() -> None:
    """The default path must be exactly what it was before any of this existed."""
    cfg = Config(flow_control=FlowControlConfig(default_credits=16))
    ctrl = AIMDFlowControl(cfg, bdp_window=None)
    assert ctrl.window == 16 and ctrl.stats()["slow_start"] == 1


def test_the_controller_says_which_constraint_is_binding() -> None:
    """A shuffle capped by memory and one capped by the wire look identical otherwise.

    They have opposite remedies — a bigger node or a narrower fan-in against a faster link —
    and nothing else in the engine can tell them apart, because nothing else knows what the
    path could have carried.
    """
    cfg = Config()
    ceiling = AIMDFlowControl(cfg).stats()["ceiling"]
    assert AIMDFlowControl(cfg, bdp_window=ceiling - 1).network_limited is True
    assert AIMDFlowControl(cfg, bdp_window=ceiling * 10).network_limited is False


def test_an_unmeasured_path_is_not_reported_as_memory_bound() -> None:
    """No evidence of a memory constraint is not the same as evidence of one."""
    assert AIMDFlowControl(Config()).network_limited is True


# --- allocating one budget across skewed channels -----------------------------------


def test_the_split_is_proportional_to_what_each_channel_carries() -> None:
    windows = proportional_windows(64, [800, 100, 100])
    assert sum(windows) == 64
    assert windows[0] > windows[1] == windows[2]


def test_an_even_split_is_worse_by_exactly_the_skew_factor() -> None:
    """The result that makes this worth doing, checked as a makespan rather than asserted.

    Channel `i` is window-limited to `w_i b / R` bytes per second, so it finishes at
    `s_i R / (w_i b)` and the reducer is done when its slowest channel is. Proportional
    allocation equalizes those times; an even split leaves the largest bucket carrying the
    whole delay. The ratio is `s_max / mean(s)` — the skew.
    """
    sizes = [1000, 100, 100, 100, 100]  # 10:1 against the mean of 280
    budget = 100

    def makespan(windows: list[int]) -> float:
        return max(s / w for s, w in zip(sizes, windows, strict=True))

    even = [budget // len(sizes)] * len(sizes)
    optimal = proportional_windows(budget, sizes)
    skew = max(sizes) / (sum(sizes) / len(sizes))
    assert makespan(optimal) < makespan(even)
    assert makespan(even) / makespan(optimal) == pytest.approx(skew, rel=0.1)


def test_no_channel_is_ever_left_unable_to_progress() -> None:
    """A zero window is not a small share; it is a channel that never completes."""
    windows = proportional_windows(8, [10_000_000, 1, 1, 1])
    assert min(windows) >= 1
    assert sum(windows) == 8


def test_a_budget_tighter_than_the_channel_count_still_feeds_everyone() -> None:
    windows = proportional_windows(2, [500, 500, 500, 500])
    assert windows == [1, 1, 1, 1], "the floor outranks the budget; a stalled reducer is worse"


def test_no_size_information_gives_an_even_split() -> None:
    """The correct answer under no information, rather than a guess."""
    assert proportional_windows(64, [0, 0, 0, 0]) == [16, 16, 16, 16]


def test_the_split_is_exact_and_deterministic() -> None:
    """Largest-remainder, ties to the earlier channel.

    Credits are integers, so rounding has to go somewhere; leaving it to chance makes a
    shuffle's timing unreproducible run to run.
    """
    sizes = [37, 41, 43, 47, 53]
    for budget in range(len(sizes), 200):
        windows = proportional_windows(budget, sizes)
        assert sum(windows) == budget, f"budget {budget} was not fully allocated"
        assert windows == proportional_windows(budget, sizes)


def test_an_empty_reducer_allocates_nothing() -> None:
    assert proportional_windows(64, []) == []


# --- trusting a learned window only as far as its own spread justifies --------------


def test_a_learned_window_past_runs_disagreed_about_keeps_its_ramp() -> None:
    """The same one-sided reasoning as the measured product, applied to history.

    A shuffle whose window has scattered over an order of magnitude has not learned a window,
    it has averaged a bimodal population. Warm-starting from that is still the best guess
    available — but switching slow start off as well leaves a badly-sized channel with no ramp
    to escape on, which is strictly worse than never having learned.
    """
    cfg = Config(flow_control=FlowControlConfig(default_credits=16))
    firm = AIMDFlowControl(cfg, initial_window=48)
    loose = AIMDFlowControl(cfg, initial_window=48, initial_window_stable=False)
    assert firm.window == loose.window == 48, "both still start from the best guess"
    assert firm.stats()["slow_start"] == 0
    assert loose.stats()["slow_start"] == 1, "an unfirm estimate must keep the search"


def test_the_agreement_behind_a_learned_window_is_asked_separately() -> None:
    """The value and the confidence have different consumers, so they are read separately."""
    from batcher.carbonite.policies import (
        load_shuffle_window,
        record_shuffle_window,
        shuffle_window_is_stable,
    )
    from batcher.metadata import MetadataHub
    from batcher.metadata.backends.in_process import InProcessBackend

    hub = MetadataHub(InProcessBackend())
    assert load_shuffle_window(hub, "unseen") is None
    assert not shuffle_window_is_stable(hub, "unseen")

    for _ in range(8):
        record_shuffle_window(hub, "steady", 32)
    assert load_shuffle_window(hub, "steady") == 32
    assert shuffle_window_is_stable(hub, "steady")

    for window in (2, 60, 4, 55, 3, 58, 5, 62):
        record_shuffle_window(hub, "erratic", window)
    assert shuffle_window_is_stable(hub, "erratic") is False
