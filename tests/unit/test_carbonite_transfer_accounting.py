"""What the shuffle measured about itself, and whether the counters survive concurrency.

A join reducer gathers its two sides on two threads and a mapper publishes while both run,
so every counter here is written from several threads at once. An unguarded `+=` on the
locality pair is the worst kind of bug this subsystem has: it under-reports the ratio that
is supposed to *prove* placement is working, and a slightly-low ratio reads as a slightly
worse shuffle rather than as a broken counter.
"""

from __future__ import annotations

import threading

import pyarrow as pa
import pytest

from batcher.carbonite.transfer import ShuffleSession, ShuffleTicket
from batcher.carbonite.transfer import server as server_mod
from batcher.carbonite.transfer.lifecycle import host_of, process_client
from batcher.carbonite.transfer.locality import TransferMode, locality_ratio

pytestmark = pytest.mark.unit


def _batch(n: int = 16) -> pa.RecordBatch:
    return pa.record_batch({"v": pa.array(list(range(n)), type=pa.int64())})


def test_transfer_modes_carry_their_own_cost_order() -> None:
    """The enum declared itself cheapest-first but carried no order to compare with."""
    modes = sorted(TransferMode, key=lambda m: m.rank)
    assert modes == [
        TransferMode.DEVICE_LOCAL,
        TransferMode.DIRECT_MEMORY,
        TransferMode.DEVICE_P2P,
        TransferMode.SHARED_MEMORY,
        TransferMode.NETWORK,
    ]
    # Every member is ranked, and no two share a rank: the ordering is what three call sites
    # compare with, and a duplicate would make two modes indistinguishable to all of them.
    assert len({m.rank for m in TransferMode}) == len(list(TransferMode))
    assert TransferMode.DEVICE_LOCAL.is_local
    assert TransferMode.DIRECT_MEMORY.is_local
    assert TransferMode.DEVICE_P2P.is_local
    assert TransferMode.SHARED_MEMORY.is_local
    assert not TransferMode.NETWORK.is_local
    assert locality_ratio([TransferMode.DIRECT_MEMORY, TransferMode.NETWORK]) == 0.5
    assert locality_ratio([]) == 1.0


def test_host_identity_drops_the_port() -> None:
    assert host_of("10.0.0.7:41234") == "10.0.0.7"
    assert host_of("10.0.0.7") == "10.0.0.7"


def test_the_pooled_client_is_one_per_process() -> None:
    assert process_client() is process_client()


def test_the_byte_locality_ratio_is_weighted_not_counted() -> None:
    """Nine tiny local fetches and one huge remote one score 0.9 by count and ~0 by bytes."""
    session = ShuffleSession()
    try:
        ticket = ShuffleTicket(1, 0, 0, 0)
        session.publish(ticket, [_batch(1000)])
        assert session.stats()["bytes_published"] > 0
        # Nothing has been served locally yet, so the byte ratio is 0 while the fetch-count
        # ratio is still its empty-set 1.0 — the two genuinely measure different things.
        assert session.stats()["byte_locality_ratio"] == 0.0
        assert session.locality_ratio == 1.0

        session.fetch(session.addr, ticket)  # a same-process (DIRECT_MEMORY) read
        stats = session.stats()
        assert stats["fetches"] == 1
        assert stats["off_network_fetches"] == 1
        assert stats["locality_ratio"] == 1.0
        assert stats["byte_locality_ratio"] == pytest.approx(1.0)
    finally:
        session.clear()


def test_concurrent_fetches_do_not_lose_locality_counts() -> None:
    """The race the stats lock exists for: two gathers on two threads, one counter pair."""
    session = ShuffleSession()
    try:
        tickets = [ShuffleTicket(1, 0, i, 0) for i in range(8)]
        for t in tickets:
            session.publish(t, [_batch(8)])

        per_thread = 200

        def puller() -> None:
            for i in range(per_thread):
                session.fetch(session.addr, tickets[i % len(tickets)])

        threads = [threading.Thread(target=puller) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        stats = session.stats()
        assert stats["fetches"] == 4 * per_thread, "increments were lost to a race"
        assert stats["off_network_fetches"] == 4 * per_thread
        assert stats["locality_ratio"] == 1.0
    finally:
        session.clear()


def test_the_shuffles_resident_footprint_is_measurable() -> None:
    """The blind spot `PressureMonitor` names: a published partition is held, never reserved."""
    with ShuffleSession() as session:
        assert session.stats()["bytes_retained"] == 0
        session.publish(ShuffleTicket(5, 0, 0, 0), [_batch(10_000)])
        held = session.stats()["bytes_retained"]
        assert held > 0, "the store holds memory nothing accounts for and nothing reports"
        assert session.stats()["partitions_retained"] == 1

        session.publish(ShuffleTicket(5, 0, 1, 0), [_batch(10_000)])
        assert session.stats()["bytes_retained"] == held * 2

        session.release(ShuffleTicket(5, 0, 0, 0))
        assert session.stats()["bytes_retained"] == held
    assert session.stats()["bytes_retained"] == 0


def test_republishing_a_ticket_does_not_drift_the_footprint() -> None:
    """A recompute republishes under the same ticket; the total must not rise forever."""
    with ShuffleSession() as session:
        ticket = ShuffleTicket(6, 0, 0, 0)
        session.publish(ticket, [_batch(5_000)])
        once = session.stats()["bytes_retained"]
        session.publish(ticket, [_batch(5_000)])
        assert session.stats()["partitions_retained"] == 1
        assert session.stats()["bytes_retained"] == once, "superseded bytes stayed on the books"


def test_the_session_clears_its_buckets_on_context_exit() -> None:
    """A session holds published buckets as zero-copy pyarrow views until something drops
    them, and the interpreter-exit hook is far too late on a long-lived worker actor."""
    with ShuffleSession() as session:
        session.publish(ShuffleTicket(9, 0, 0, 0), [_batch()])
        assert session.partition_count == 1
    assert session.partition_count == 0


def test_the_ingress_counter_is_resettable_per_query() -> None:
    """A process lifetime total cannot answer "how much did *this query* fetch"."""
    server_mod.reset_bytes_fetched()
    assert server_mod.bytes_fetched() == 0
    server_mod._add_bytes_fetched(4096)
    assert server_mod.bytes_fetched() == 4096
    server_mod.reset_bytes_fetched()
    assert server_mod.bytes_fetched() == 0


def test_the_ingress_counter_survives_concurrent_writers() -> None:
    server_mod.reset_bytes_fetched()
    try:

        def add() -> None:
            for _ in range(2000):
                server_mod._add_bytes_fetched(1)

        threads = [threading.Thread(target=add) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        assert server_mod.bytes_fetched() == 8000
    finally:
        server_mod.reset_bytes_fetched()


def test_a_static_session_reports_its_credit_window() -> None:
    session = ShuffleSession(credits=12)
    try:
        assert session.stats()["credit_window"] == 12
        session.set_credits(3)
        assert session.stats()["credit_window"] == 3
    finally:
        session.clear()


def test_an_adaptive_session_reports_the_controllers_statistics() -> None:
    from batcher.carbonite.policies import AIMDFlowControl
    from batcher.config import Config

    ctrl = AIMDFlowControl(Config())
    session = ShuffleSession(flow_control=ctrl)
    try:
        ctrl.observe(congested=True)
        stats = session.stats()
        assert stats["credit_backoffs"] == 1
        assert stats["credit_rounds"] == 1
        assert "credit_window" in stats
        # A re-grant under adaptive credits must reach the controller, not a dead field.
        session.set_credits(9)
        assert ctrl.window == 9
    finally:
        session.clear()


# --- shared memory is RAM, so its lifetime is a memory bound -------------------


def _shm_root() -> str:
    import os

    return "/dev/shm/batcher_shm" if os.path.isdir("/dev/shm") else "/tmp/batcher_shm"


def _peer_dirs() -> set[str]:
    import glob
    import os

    return {d for d in glob.glob(_shm_root() + "/*") if os.path.isdir(d)}


@pytest.mark.skipif(not hasattr(ShuffleSession, "publish"), reason="transport not built")
def test_a_dropped_session_frees_its_shared_memory_within_a_live_process() -> None:
    """The startup reaper cannot cover this case, by design.

    Shm buckets live in tmpfs, which is RAM. A worker that dies without teardown is
    handled by the reaper a later process runs — but that reaper deliberately spares
    directories owned by the *current* pid, because it cannot distinguish a dead session's
    from a live one's and reaping a live peer's buckets would be far worse than leaking.
    So a long-lived worker creating and dropping sessions (the session-fleet shape)
    accumulated its own dead sessions' shm until the process exited. The server's `Drop`
    is the same-process half of the pair.
    """
    import gc

    before = _peer_dirs()
    session = ShuffleSession(shm=True)
    session.publish(ShuffleTicket(41, 0, 0, 0), [_batch(5_000)])
    created = _peer_dirs() - before
    if not created:
        pytest.skip("no shared-memory directory available on this host")

    # Dropped *without* calling clear(), in a process that keeps running.
    del session
    gc.collect()

    assert not (created & _peer_dirs()), (
        "a dropped session left its buckets in tmpfs — that is RAM the process never gets back"
    )
