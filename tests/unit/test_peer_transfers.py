"""Per-peer shuffle throughput, and the refusals that keep a healthy node from being drained.

A straggler diagnosis that fires on noise is worse than none: it sends an operator to inspect
a node that was fine. Every case here is about when the answer is `None`.
"""

from __future__ import annotations

import pytest

from batcher.carbonite.transfer.peers import PeerTransfer, peer_transfers, straggler_peer

pytestmark = pytest.mark.unit

MB = 1024 * 1024


def _peer(addr: str, gbps: float, *, mb: int = 64) -> PeerTransfer:
    """A record whose rate is exactly `gbps`."""
    nbytes = mb * MB
    return PeerTransfer(addr=addr, bytes=nbytes, seconds=nbytes * 8 / 1e9 / gbps, fetches=4)


def test_the_rate_is_bits_over_seconds() -> None:
    assert PeerTransfer("h:1", bytes=1_000_000_000, seconds=1.0).gbps == pytest.approx(8.0)


def test_an_unmeasured_peer_has_no_rate_rather_than_a_zero_one() -> None:
    assert PeerTransfer("h:1", bytes=1_000).gbps == 0.0


def test_a_peer_far_below_the_fleet_median_is_named() -> None:
    fleet = (_peer("a", 100.0), _peer("b", 100.0), _peer("c", 100.0), _peer("slow", 10.0))
    found = straggler_peer(fleet)
    assert found is not None
    assert found.addr == "slow"


def test_an_even_fleet_names_nobody() -> None:
    fleet = (_peer("a", 100.0), _peer("b", 90.0), _peer("c", 95.0), _peer("d", 88.0))
    assert straggler_peer(fleet) is None


def test_the_median_is_used_so_one_slow_peer_cannot_hide_behind_the_mean() -> None:
    """A mean is dragged toward the outlier; the median is not."""
    fleet = (_peer("a", 100.0), _peer("b", 100.0), _peer("c", 100.0), _peer("slow", 1.0))
    assert straggler_peer(fleet).addr == "slow"


def test_a_fleet_of_two_is_not_a_straggler_diagnosis() -> None:
    assert straggler_peer((_peer("a", 100.0), _peer("slow", 1.0))) is None


def test_a_peer_that_barely_moved_anything_is_noise_not_a_straggler() -> None:
    """A single small fetch measures the connection setup rather than the wire."""
    tiny = PeerTransfer(addr="tiny", bytes=1024, seconds=1.0, fetches=1)
    fleet = (_peer("a", 100.0), _peer("b", 100.0), _peer("c", 100.0), tiny)
    assert straggler_peer(fleet) is None


def test_an_unmeasured_fleet_names_nobody() -> None:
    assert straggler_peer(()) is None
    assert straggler_peer((PeerTransfer("a"), PeerTransfer("b"), PeerTransfer("c"))) is None


def test_the_threshold_is_tunable_for_a_stricter_check() -> None:
    fleet = (_peer("a", 100.0), _peer("b", 100.0), _peer("c", 100.0), _peer("d", 60.0))
    assert straggler_peer(fleet) is None
    assert straggler_peer(fleet, ratio=0.8).addr == "d"


def test_reading_the_counters_never_raises_on_an_older_engine() -> None:
    """A worker whose extension predates the counters keeps the statistics it had."""
    assert isinstance(peer_transfers(), tuple)
