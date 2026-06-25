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


def test_shared_memory_fetch_skips_the_network():
    """Two same-node sessions in *different* objects (simulating different processes):
    with shared memory on, the mapper mirrors its bucket to an mmap'd file and the
    reducer reads it via SHARED_MEMORY — off the network, no gRPC. Data is identical."""
    mapper = ShuffleSession(shm=True)
    reducer = ShuffleSession(shm=True)
    ticket = ShuffleTicket(5, 0, 0, 1)
    mapper.publish(ticket, [pa.record_batch({"k": [1, 2, 3]}), pa.record_batch({"k": [4]})])
    try:
        got = reducer.fetch(mapper.addr, ticket)
        assert _keys(got) == [1, 2, 3, 4]
        # Same node, different address ⇒ the fetch went through shared memory, not a socket.
        assert reducer.locality_ratio == 1.0
    finally:
        mapper.clear()


def test_shared_memory_miss_falls_back_to_flight():
    """A SHARED_MEMORY-eligible fetch whose bucket was never shm'd (an empty bucket, or
    the producer had shm off) transparently falls back to Flight — still correct."""
    mapper = ShuffleSession(shm=False)  # publishes to Flight only, no shm file
    reducer = ShuffleSession(shm=True)
    ticket = ShuffleTicket(6, 0, 0, 1)
    mapper.publish(ticket, [pa.record_batch({"k": [10, 20]})])
    try:
        got = reducer.fetch(mapper.addr, ticket)  # shm miss → Flight
        assert _keys(got) == [10, 20]
        assert reducer.locality_ratio == 0.0  # the shm miss went over the network
    finally:
        mapper.clear()


def test_shared_memory_off_is_unchanged_network_behavior():
    """With shared memory off (the default), a same-node remote fetch is NETWORK exactly
    as before — no shm files, no behavior change."""
    mapper = ShuffleSession()  # shm off (default)
    reducer = ShuffleSession()
    ticket = ShuffleTicket(7, 0, 0, 1)
    mapper.publish(ticket, [pa.record_batch({"k": [99]})])
    got = reducer.fetch(mapper.addr, ticket)
    assert _keys(got) == [99]
    assert reducer.locality_ratio == 0.0  # NETWORK (default path unchanged)


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


def test_gather_concat_collects_remote_and_local_sources():
    """`gather_concat` fetches every source concurrently — remote *and* the reducer's
    own co-located bucket (DIRECT_MEMORY) — and returns no lost sources."""
    p1, p2 = ShuffleSession(), ShuffleSession()
    reducer = ShuffleSession()
    p1.publish(ShuffleTicket(7, 0, 0, 1), [pa.record_batch({"k": [1, 2]})])
    p2.publish(ShuffleTicket(7, 0, 1, 1), [pa.record_batch({"k": [3]})])
    reducer.publish(ShuffleTicket(7, 0, 2, 1), [pa.record_batch({"k": [4, 5]})])  # co-located

    rows, unreachable = reducer.gather_concat(
        [
            (p1.addr, ShuffleTicket(7, 0, 0, 1)),
            (p2.addr, ShuffleTicket(7, 0, 1, 1)),
            (reducer.addr, ShuffleTicket(7, 0, 2, 1)),  # own server → no socket
        ]
    )
    assert unreachable == []
    assert _keys(rows) == [1, 2, 3, 4, 5]


def test_gather_concat_reports_unreachable_source():
    """An unreachable peer is a *retryable* fault — reported as its source index
    (the `("retry", srcs)` signal), never silently dropped to an empty bucket."""
    producer = ShuffleSession()
    reducer = ShuffleSession()
    producer.publish(ShuffleTicket(8, 0, 0, 0), [pa.record_batch({"k": [1]})])

    _rows, unreachable = reducer.gather_concat(
        [
            (producer.addr, ShuffleTicket(8, 0, 0, 0)),  # source 0: reachable
            ("127.0.0.1:1", ShuffleTicket(8, 0, 1, 0)),  # source 1: dead port
        ]
    )
    assert unreachable == [1]  # the dead source's index, for driver recompute + retry


def test_gather_combine_matches_serial_combine_finalize():
    """`gather_combine` fetches partials concurrently and folds them in Rust; the
    result equals a serial `combine_finalize` (combine is associative+commutative)."""
    import batcher._native as nat

    import batcher as bt
    from batcher import col, count

    rng = np.random.default_rng(11)
    n = 30_000
    t = pa.table(
        {"k": rng.integers(0, 40, n).astype("int64"), "v": rng.integers(0, 100, n).astype("int64")}
    )
    ds = bt.from_arrow(t).group_by("k").agg(s=col("v").sum(), c=count(), a=col("v").mean())
    agg = ds._plan
    import json as _json

    gk = _json.dumps([{"expr": k.expr.to_ir(), "alias": k.alias} for k in agg.group_keys])
    aj = _json.dumps([s.agg.to_ir(s.alias) for s in agg.aggregates])

    # One partial per producer session = a wide fan-in of mappers feeding one reducer.
    chunks = list(t.to_batches(max_chunksize=3000))
    producers = [ShuffleSession() for _ in chunks]
    sources = []
    for i, (p, chunk) in enumerate(zip(producers, chunks, strict=True)):
        partial = nat.partial_aggregate(gk, aj, [chunk])
        ticket = ShuffleTicket(10, 0, i, 0)
        p.publish(ticket, [partial])
        sources.append((p.addr, ticket))

    reducer = ShuffleSession()
    payload, unreachable = reducer.gather_combine(gk, aj, sources, finalize=True)
    assert unreachable == []

    def _rows(table):
        return sorted(
            tuple(round(v, 6) if isinstance(v, float) else v for v in r.values())
            for r in table.to_pylist()
        )

    assert _rows(pa.Table.from_batches([payload])) == _rows(ds.collect())
