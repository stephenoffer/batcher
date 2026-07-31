"""A shard that overflowed is divided by how far it went over, not by a blind halving.

`measured_parts` reads a device high-water mark and divides an over-large shard by the factor
that clears it in one round. It read that mark from **its own process** — and the subdivision
is decided on the driver, which has no device. So on every distributed run the measurement
returned nothing and the division silently fell back to halving: a shard eight times too large
took three failed rounds to find a size that fits, and each of those rounds re-read the whole
shard from storage. Every gate passed the entire time, because the fallback is correct.

The fix carries the figure inside the error, which is the one channel a task failure is
guaranteed to travel through intact — Ray re-raises a task's exception as its own wrapper type
and preserves the message. These cover both halves: that a worker's marker is produced and
parsed, and that the absence of one keeps the previous blind behavior exactly.
"""

from __future__ import annotations

import pytest

from batcher.dist.gpu.shards import (
    MAX_MEASURED_PARTS,
    is_memory_failure,
    measured_parts,
    peak_from_error,
    run_subdivided,
)

pytestmark = pytest.mark.unit


def _overflow(peak: int, pool: int) -> MemoryError:
    """The error a GPU worker raises for an overflow it measured."""
    return MemoryError(
        f"RuntimeError: std::bad_alloc: out_of_memory [bt-device-peak {peak}/{pool}]"
    )


def test_a_workers_marker_is_read_back_off_the_error() -> None:
    assert peak_from_error(_overflow(8_000, 1_000)) == (8_000, 1_000)


def test_a_marker_survives_being_chained_under_another_error() -> None:
    # Ray re-raises through a wrapper, and a fallback path may chain again on the way up.
    try:
        raise _overflow(4_000, 1_000)
    except MemoryError as inner:
        outer = RuntimeError("ray task failed")
        outer.__cause__ = inner
        assert peak_from_error(outer) == (4_000, 1_000)


def test_an_error_with_no_marker_reports_nothing_rather_than_a_guess() -> None:
    assert peak_from_error(MemoryError("out of memory")) == (0, 0)
    assert peak_from_error(None) == (0, 0)
    assert peak_from_error(ValueError("no column 'x'")) == (0, 0)


def test_the_division_clears_the_overflow_in_one_round() -> None:
    # Eight times over the pool is divided by eight, not halved three times over three reads.
    assert measured_parts(2, _overflow(8_000, 1_000)) == 8
    assert measured_parts(2, _overflow(2_500, 1_000)) == 3, "rounded up: 2.5x needs 3 pieces"


def test_the_division_never_falls_below_the_callers_own_factor() -> None:
    assert measured_parts(4, _overflow(2_000, 1_000)) == 4


def test_a_runaway_measurement_is_capped_rather_than_believed() -> None:
    assert measured_parts(2, _overflow(10**9, 1)) == MAX_MEASURED_PARTS


def test_a_shard_that_fits_is_not_divided_by_a_fraction() -> None:
    assert measured_parts(2, _overflow(500, 1_000)) == 2, "peak below the pool measures nothing"


def test_no_marker_keeps_the_blind_halving_the_driver_always_had() -> None:
    assert measured_parts(2, MemoryError("out of memory")) == 2
    assert measured_parts() == 2


def test_a_measured_error_still_classifies_as_a_memory_failure() -> None:
    # The marker must not change what the failure *is*: a shard whose overflow stopped reading
    # as a memory failure would skip the subdivision ladder entirely and go straight to the CPU.
    assert is_memory_failure(_overflow(8_000, 1_000))


def test_the_first_division_is_sized_from_the_error_that_caused_it() -> None:
    """The whole point: the driver divides by what the *worker* measured."""
    seen: list[int] = []

    def _run(descriptor):
        seen.append(len(descriptor["splits"]))
        return None

    run_subdivided(
        {"splits": list(range(64))},
        _run,
        parts=2,
        rounds=1,
        cause=_overflow(8_000, 1_000),
    )
    assert seen == [8] * 8, "64 splits divided by the measured 8, not the default 2"


def test_with_no_cause_the_division_is_the_configured_default() -> None:
    seen: list[int] = []

    def _run(descriptor):
        seen.append(len(descriptor["splits"]))
        return None

    run_subdivided({"splits": list(range(64))}, _run, parts=2, rounds=1)
    assert seen == [32, 32]


def test_a_piece_that_overflows_again_is_re_measured_from_its_own_failure() -> None:
    """Each round divides by what that round measured, not by what the first one did."""
    widths: list[int] = []
    failed_once = {"done": False}

    def _run(descriptor):
        widths.append(len(descriptor["splits"]))
        if not failed_once["done"]:
            failed_once["done"] = True
            raise _overflow(4_000, 1_000)
        return None

    run_subdivided({"splits": list(range(64))}, _run, parts=2, rounds=3)
    # First division by the caller's default (no cause), then the failed 32-split piece is
    # divided by the 4x its own error reported.
    assert widths[0] == 32
    assert 8 in widths, f"the failed piece should divide by 4, giving 8-split pieces: {widths}"
