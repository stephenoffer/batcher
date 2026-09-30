"""A bucket is only mirrored into shared memory when something could read it.

`shared_memory_transfer` mirrors every published bucket into a tmpfs Arrow IPC file so a
same-node reducer *in another process* can mmap it instead of taking a gRPC hop. That pays
on a fleet that packs several workers per node. It pays nothing on a fleet of small nodes,
where `dist.executor._numa_sliced` puts one worker on a node: a same-address fetch is served
from the local store and every other fetch is on another machine, so the mirror is a second
serialization of every bucket plus an `unlink`, for no reader at all.

The fleet is the only thing that can know: a shuffle address is `{node_ip}:{port}`, so two
workers share a node exactly when their addresses' hosts match.
"""

from __future__ import annotations

import pytest

from batcher.carbonite.transfer.lifecycle import host_of


def _real_session(shm=True):
    from batcher.carbonite.transfer.session import ShuffleSession

    s = object.__new__(ShuffleSession)
    s._shm = shm
    s._shm_peers = True
    s._pressure = None
    return s


def test_a_session_mirrors_by_default():
    """The default has to be "assume a peer": any path that never sets the flag must behave
    exactly as it did before this gate existed."""
    assert _real_session()._shm_mirror_ok() is True


def test_a_session_told_it_is_alone_does_not_mirror():
    s = _real_session()
    s.set_shm_peers(False)
    assert s._shm_mirror_ok() is False


def test_being_told_it_has_a_peer_restores_the_mirror():
    s = _real_session()
    s.set_shm_peers(False)
    s.set_shm_peers(True)
    assert s._shm_mirror_ok() is True


@pytest.mark.parametrize(
    ("addrs", "expected"),
    [
        # One worker per node: nobody has a peer. The shape this exists for.
        (["10.0.0.1:40000", "10.0.0.2:40000", "10.0.0.3:40000"], [False, False, False]),
        # Several workers per node: everybody does.
        (["10.0.0.1:40000", "10.0.0.1:40001"], [True, True]),
        # Mixed, which is the case a "all or nothing" shortcut would get wrong.
        (
            ["10.0.0.1:40000", "10.0.0.1:40001", "10.0.0.2:40000"],
            [True, True, False],
        ),
        (["10.0.0.1:40000"], [False]),
    ],
)
def test_peers_are_decided_by_the_address_host(addrs, expected):
    """The arithmetic the fleet applies, held separately from the Ray plumbing around it."""
    hosts = [host_of(a) for a in addrs]
    counts: dict[str, int] = {}
    for h in hosts:
        counts[h] = counts.get(h, 0) + 1
    assert [counts[h] > 1 for h in hosts] == expected


def test_the_fleet_tells_every_worker_and_survives_one_that_cannot_be_told():
    """One round-trip per fleet, and a failure leaves the default rather than raising.

    The helper is best-effort on purpose: a missing mirror already falls back to Flight, so
    the worst case of not being told is the behaviour that shipped before.
    """
    from batcher.dist.fleet.spawn import _tell_workers_about_node_peers

    told: list[bool] = []

    class _Ref:
        def __init__(self, value):
            self.value = value

    class _Method:
        def __init__(self, sink):
            self.sink = sink

        def remote(self, has_peer):
            self.sink.append(has_peer)
            return _Ref(has_peer)

    class _Actor:
        def __init__(self):
            self.set_shm_peers = _Method(told)

    actors = [_Actor(), _Actor(), _Actor()]
    import sys
    import types

    fake = types.ModuleType("ray")
    fake.get = lambda refs: [r.value for r in refs]
    saved = sys.modules.get("ray")
    sys.modules["ray"] = fake
    try:
        _tell_workers_about_node_peers(actors, ["10.0.0.1:1", "10.0.0.1:2", "10.0.0.9:1"])
        assert told == [True, True, False]
        told.clear()
        # A mismatched actor/address count must not propagate. `zip(strict=True)` raises
        # part-way through, so *some* workers may already have been told — what matters is
        # that the fleet keeps running and the untold ones keep the safe default, not that
        # nobody was told.
        _tell_workers_about_node_peers(actors, ["10.0.0.1:1"])
        assert len(told) < len(actors), "a mismatch should not have reached every worker"
    finally:
        if saved is not None:
            sys.modules["ray"] = saved
        else:
            del sys.modules["ray"]


def test_the_mirror_is_counted_so_capacity_planning_can_see_it():
    """The tmpfs mirror is a second resident copy the buffer pool and the store cap never
    see, so the session reports its volume, and the buckets whose mirror was skipped."""
    import threading

    import pyarrow as pa

    class _Server:
        def __init__(self):
            self.shared = []

        def publish(self, ticket, batches):
            return sum(b.nbytes for b in batches)

        def publish_shared(self, ticket, batches):
            self.shared.append(ticket)

    s = _real_session()
    s._server = _Server()
    s._stats_lock = threading.Lock()
    s._shm_mirrored_bytes = 0
    s._shm_mirrors_skipped = 0
    batch = pa.record_batch({"x": pa.array(range(1000), pa.int64())})

    s.publish("t0", [batch])
    assert s._server.shared == ["t0"]
    assert s._shm_mirrored_bytes == batch.nbytes

    s.set_shm_peers(False)
    s.publish("t1", [batch])
    assert s._server.shared == ["t0"], "a skipped mirror must not be written"
    assert s._shm_mirrored_bytes == batch.nbytes
    assert s._shm_mirrors_skipped == 1
