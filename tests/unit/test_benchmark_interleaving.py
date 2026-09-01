"""A sweep that times variants in blocks reads drift as a difference between them.

This is the failure `benchmarks/harness/interleave.py` exists for, and it is not a
hypothetical: sweeping an aggregate's reducer count in the fixed order [2, 4, 8, 16, 32, 64]
produced a clean interior optimum at 8 across four cardinalities, which was written up and
used to derive a mechanism before a round-robin re-run showed the curve was monotone and the
"optimum" was 20% worse than its neighbour. The whole shape was a fleet warming up.

These tests reproduce that with a clock that only ever drifts downward — no variant is
genuinely faster than any other — and pin that block ordering invents a winner while
interleaving does not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
if str(_BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS))

from harness.interleave import interleaved, overlapping  # noqa: E402

pytestmark = pytest.mark.unit

_VARIANTS = ("a", "b", "c", "d")


class _DriftingClock:
    """A monotonically improving environment: every call is a little faster than the last.

    Stands in for a warming fleet or a filling page cache. Crucially the *variant is never
    read*, so any difference a sweep reports between variants is manufactured entirely by
    when each one ran.
    """

    def __init__(self, start: float = 100.0, step: float = 3.0) -> None:
        self.now = 0.0
        self._cost = start
        self._step = step

    def run(self, _variant: object) -> None:
        self.now += self._cost
        self._cost = max(1.0, self._cost - self._step)


def _blocked(variants, clock, rounds):
    """The sweep shape this module exists to replace: all reps of one variant, then the next."""
    out = {}
    for v in variants:
        clock.run(v)  # the same warmup `interleaved` does
    for v in variants:
        times = []
        for _ in range(rounds):
            before = clock.now
            clock.run(v)
            times.append(clock.now - before)
        out[v] = times
    return out


def _fake_perf_counter(clock, monkeypatch):
    """Point `interleave`'s clock at the drifting one, in seconds."""
    monkeypatch.setattr(
        "harness.interleave.time.perf_counter", lambda: clock.now / 1000.0, raising=True
    )


def test_block_ordering_reports_a_winner_that_does_not_exist(monkeypatch):
    """The control: with no real difference, a blocked sweep still ranks the variants."""
    clock = _DriftingClock()
    blocked = _blocked(_VARIANTS, clock, rounds=3)
    medians = {v: sorted(t)[len(t) // 2] for v, t in blocked.items()}
    best, worst = min(medians, key=medians.get), max(medians, key=medians.get)
    assert best != worst, "the drift must produce *some* ranking, or this control is vacuous"
    assert medians[best] < medians[worst] * 0.9, (
        f"blocked ordering separated identical variants by <10%: {medians} — the fixture's "
        "drift is too weak to demonstrate the artefact"
    )
    assert best == _VARIANTS[-1], "the drift is downward, so the last block must look fastest"


def test_interleaving_does_not_separate_identical_variants(monkeypatch):
    """The property: the same drift, round-robin, leaves every variant's range overlapping."""
    clock = _DriftingClock()
    _fake_perf_counter(clock, monkeypatch)
    timings = interleaved(_VARIANTS, clock.run, rounds=3)
    assert set(timings) == set(_VARIANTS)
    assert all(len(t) == 3 for t in timings.values())
    for i, a in enumerate(_VARIANTS):
        for b in _VARIANTS[i + 1 :]:
            assert overlapping(timings[a], timings[b]), (
                f"{a} and {b} were separated by drift alone: {timings[a]} vs {timings[b]}"
            )


def test_interleaving_still_sees_a_real_difference(monkeypatch):
    """The other half: a variant that is genuinely slower must still be resolved.

    Without this, a helper that reported "no difference" unconditionally would pass the test
    above — the shape `just lint-tests` calls an assertion true by construction.
    """
    clock = _DriftingClock()
    _fake_perf_counter(clock, monkeypatch)

    def run(variant):
        clock.run(variant)
        if variant == "d":
            clock.now += 500.0  # far larger than the drift across the whole sweep

    timings = interleaved(_VARIANTS, run, rounds=3)
    assert not overlapping(timings["d"], timings["a"]), (
        f"a 500ms penalty was lost in the noise: {timings}"
    )
    assert min(timings["d"]) > max(timings["a"])


def test_every_variant_is_warmed_before_any_is_timed(monkeypatch):
    """A first-call cost must not be charged to whichever variant ran first."""
    clock = _DriftingClock()
    _fake_perf_counter(clock, monkeypatch)
    calls: list[object] = []

    def run(variant):
        calls.append(variant)
        clock.run(variant)

    interleaved(_VARIANTS, run, rounds=2)
    assert calls[: len(_VARIANTS)] == list(_VARIANTS), "the warmup pass must precede the rounds"
    assert len(calls) == len(_VARIANTS) * 3, "one warmup plus two timed rounds per variant"


def test_warmup_can_be_declined(monkeypatch):
    clock = _DriftingClock()
    _fake_perf_counter(clock, monkeypatch)
    calls: list[object] = []
    interleaved(_VARIANTS, lambda v: (calls.append(v), clock.run(v)), rounds=2, warmup=False)
    assert len(calls) == len(_VARIANTS) * 2


def test_rounds_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        interleaved(("a",), lambda _v: None, rounds=0)


def test_overlapping_is_conservative_about_empty_input():
    """No data cannot separate anything, so it must not claim to."""
    assert overlapping([], [1.0])
    assert overlapping([1.0], [])
