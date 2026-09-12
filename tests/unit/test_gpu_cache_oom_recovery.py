"""On a device out-of-memory, give back the cache before asking the driver to subdivide.

The device frame cache is the one part of a GPU worker's footprint that is pure optimization:
every byte of it is a shard that could be read again. Device memory is also the resource a GPU
query actually runs out of, so the cache is exactly what stands between a shard fitting and not.

The driver's answer to an overflow is `dist.gpu.shards.run_subdivided` — divide the shard and
**re-read each piece from storage**, up to `gpu_shard_subdivide_rounds` times. Paying that
because a neighbour's cached shard was in the way is the wrong trade by a wide margin, and the
process that overflowed is the only one that can release it.

So `_measured` drops the cache and retries once, and only when there was something to drop —
otherwise a genuine overflow would run the same failing shard twice before reporting it.
"""

from __future__ import annotations

import pytest

from batcher.dist.gpu import tasks

pytestmark = pytest.mark.unit


class _Oom(MemoryError):
    pass


@pytest.fixture
def released(monkeypatch):
    """Control what the cache says it released, and count the clears."""
    state = {"bytes": 0, "clears": 0}

    def _clear():
        state["clears"] += 1
        return state["bytes"]

    monkeypatch.setattr("batcher.dist.gpu.device_read.clear_device_frame_cache", _clear)
    monkeypatch.setattr("batcher.dist.gpu.shards.device_peak_marker", lambda: "")
    return state


def test_a_holding_cache_is_released_and_the_shard_retried(released):
    released["bytes"] = 4096
    attempts = []

    def _run():
        attempts.append(1)
        if len(attempts) == 1:
            raise _Oom("out of memory")
        return "ok"

    assert tasks._measured(_run) == "ok"
    assert released["clears"] == 1
    assert len(attempts) == 2


def test_an_empty_cache_does_not_buy_a_second_attempt(released):
    """Nothing was released, so nothing changed — running the same failing shard again would
    only double the time before the driver hears about it."""
    released["bytes"] = 0
    attempts = []

    def _run():
        attempts.append(1)
        raise _Oom("out of memory")

    with pytest.raises(MemoryError):
        tasks._measured(_run)
    assert len(attempts) == 1


def test_a_shard_that_still_does_not_fit_reports_the_second_failure(released):
    released["bytes"] = 4096
    attempts = []

    def _run():
        attempts.append(1)
        raise _Oom(f"attempt {len(attempts)}")

    with pytest.raises(MemoryError, match="attempt 2"):
        tasks._measured(_run)
    assert len(attempts) == 2


def test_a_non_memory_failure_never_touches_the_cache(released):
    """An untranslatable expression fails identically on a smaller shard and with a cold cache;
    dropping the cache for it would throw away the whole fan-out's warmth for nothing."""
    released["bytes"] = 4096

    def _run():
        raise ValueError("column 'x' absent from the GPU frame")

    with pytest.raises(ValueError):
        tasks._measured(_run)
    assert released["clears"] == 0


def test_a_successful_shard_never_touches_the_cache(released):
    released["bytes"] = 4096
    assert tasks._measured(lambda: "ok") == "ok"
    assert released["clears"] == 0


def test_the_measured_peak_still_reaches_the_driver(released, monkeypatch):
    """The subdivision that follows an overflow is sized from the high-water mark, and the
    figure exists only in the process that overflowed."""
    released["bytes"] = 0
    monkeypatch.setattr("batcher.dist.gpu.shards.device_peak_marker", lambda: " [peak=7GiB]")

    def _run():
        raise _Oom("out of memory")

    with pytest.raises(MemoryError, match=r"peak=7GiB"):
        tasks._measured(_run)
