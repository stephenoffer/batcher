"""Carbonite ShuffleSession as a standalone, locality-aware transfer engine.

These tests drive the session with no Ray, no `dist`, no optimizer/executor — just
N in-process sessions publishing and gathering — proving Carbonite is a transfer
sublibrary on its own. They pin the two governors: a co-located bucket takes the
`DIRECT_MEMORY` path (no socket), and a credit window bounds a remote stream.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from batcher.carbonite.transfer import ShuffleSession, ShuffleTicket, TransferMode, select_mode

pytest.importorskip("batcher._native", reason="native engine not built")


def _keys(batches):
    return sorted(int(k) for b in batches for k in b.column("k").to_pylist())


def test_pooled_client_bounds_connections_to_peers():
    """The pooled consumer reuses one gRPC channel per peer.

    Fetching a peer many times is a single connection, so an all-to-all shuffle
    over P peers costs O(P) connections, not O(partitions) — the property that lets
    the shuffle scale to a very large cluster.
    """
    from batcher.carbonite.transfer.server import ShuffleClient

    producers = [ShuffleSession() for _ in range(3)]
    for i, p in enumerate(producers):
        for d in range(4):  # each peer hosts several partitions
            p.publish(ShuffleTicket(9, 0, i, d), [pa.record_batch({"k": [i * 10 + d]})])

    client = ShuffleClient()
    for i, p in enumerate(producers):  # 12 fetches across 3 peers
        for d in range(4):
            client.fetch(p.addr, ShuffleTicket(9, 0, i, d), 2)
    assert client.connection_count == 3  # one channel per peer, not per fetch


def test_mode_selection_is_pure():
    assert select_mode("h:1", "h:1") is TransferMode.DIRECT_MEMORY
    assert select_mode("h:1", "h:2") is TransferMode.NETWORK
    assert select_mode("h:1", "h:2", source_node="n", local_node="n") is TransferMode.SHARED_MEMORY


def test_two_session_shuffle_roundtrip():
    """Two producer sessions publish buckets; a third reducer gathers from both."""
    p1, p2, reducer = ShuffleSession(), ShuffleSession(), ShuffleSession()
    p1.publish(ShuffleTicket(1, 0, 0, 2), [pa.record_batch({"k": [1, 2, 3]})])
    p2.publish(ShuffleTicket(1, 0, 1, 2), [pa.record_batch({"k": [4, 5]})])

    got = reducer.gather(
        [(p1.addr, ShuffleTicket(1, 0, 0, 2)), (p2.addr, ShuffleTicket(1, 0, 1, 2))]
    )
    assert _keys(got) == [1, 2, 3, 4, 5]
    # Both sources were remote → fully on the network.
    assert reducer.locality_ratio == 0.0


def test_same_session_fetch_is_direct_memory():
    """A session that publishes and fetches its own bucket skips the network."""
    s = ShuffleSession()
    ticket = ShuffleTicket(2, 0, 0, 0)
    s.publish(ticket, [pa.record_batch({"k": [7, 8, 9]})])

    got = s.fetch(s.addr, ticket)
    assert _keys(got) == [7, 8, 9]
    assert s.locality_ratio == 1.0  # the only fetch never hit a socket


def test_mixed_local_and_remote_gather():
    """A reducer co-located with one producer: that bucket is direct, the other net."""
    remote = ShuffleSession()
    reducer = ShuffleSession()
    # reducer also acts as a producer for its own bucket (the self-shuffle case).
    reducer.publish(ShuffleTicket(3, 0, 1, 0), [pa.record_batch({"k": [10, 11]})])
    remote.publish(ShuffleTicket(3, 0, 0, 0), [pa.record_batch({"k": [20]})])

    got = reducer.gather(
        [
            (remote.addr, ShuffleTicket(3, 0, 0, 0)),  # NETWORK
            (reducer.addr, ShuffleTicket(3, 0, 1, 0)),  # DIRECT_MEMORY
        ]
    )
    assert _keys(got) == [10, 11, 20]
    assert reducer.locality_ratio == 0.5  # one of two fetches stayed off the network


def test_credit_window_bounds_remote_stream():
    """A small granted window bounds the producer's in-flight high-water mark."""
    producer = ShuffleSession()
    reducer = ShuffleSession(credits=2)
    ticket = ShuffleTicket(5, 0, 0, 0)
    rng = np.random.default_rng(0)
    batches = [pa.record_batch({"k": rng.integers(0, 100, 100)}) for _ in range(20)]
    producer.publish(ticket, batches)

    got = reducer.fetch(producer.addr, ticket)
    assert sum(b.num_rows for b in got) == sum(b.num_rows for b in batches)
    hw = producer.max_inflight(ticket)
    assert hw is not None and 1 <= hw <= 2


def test_gather_tolerates_missing_buckets():
    """A mapper with no rows for this bucket (unpublished ticket) is just skipped."""
    producer = ShuffleSession()
    reducer = ShuffleSession()
    producer.publish(ShuffleTicket(6, 0, 0, 0), [pa.record_batch({"k": [1]})])
    got = reducer.gather(
        [
            (producer.addr, ShuffleTicket(6, 0, 0, 0)),  # exists
            (producer.addr, ShuffleTicket(6, 0, 9, 0)),  # never published → empty
        ]
    )
    assert _keys(got) == [1]
