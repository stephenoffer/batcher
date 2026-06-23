"""Arrow Flight shuffle transport: multi-node data plane, object store bypassed."""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from batcher.carbonite.transfer import FlightShuffleServer, ShuffleTicket, fetch


def _keys(batches):
    return sorted(int(k) for b in batches for k in b.column("k").to_pylist())


def test_two_node_shuffle_roundtrip():
    """Two mapper nodes publish partitions; a reducer fetches from both."""
    n1, n2 = FlightShuffleServer(), FlightShuffleServer()
    n1.publish(ShuffleTicket(1, 0, 0, 2), [pa.record_batch({"k": [1, 2, 3], "v": [10, 20, 30]})])
    n2.publish(ShuffleTicket(1, 0, 1, 2), [pa.record_batch({"k": [4, 5], "v": [40, 50]})])

    got = fetch(n1.addr, ShuffleTicket(1, 0, 0, 2)) + fetch(n2.addr, ShuffleTicket(1, 0, 1, 2))
    assert _keys(got) == [1, 2, 3, 4, 5]


def test_many_batch_credit_bounded_stream():
    """A large multi-batch partition streams correctly under credit flow control."""
    srv = FlightShuffleServer()
    rng = np.random.default_rng(0)
    batches = [
        pa.record_batch({"k": rng.integers(0, 1000, 500), "v": rng.integers(0, 100, 500)})
        for _ in range(40)
    ]
    srv.publish(ShuffleTicket(9, 1, 0, 0), batches)

    got = fetch(srv.addr, ShuffleTicket(9, 1, 0, 0))
    expected = sum(b.num_rows for b in batches)
    assert sum(b.num_rows for b in got) == expected


def test_distinct_tickets_isolated():
    """Different destination partitions on one server are independent."""
    srv = FlightShuffleServer()
    srv.publish(ShuffleTicket(1, 0, 0, 0), [pa.record_batch({"k": [1, 1], "v": [1, 1]})])
    srv.publish(ShuffleTicket(1, 0, 0, 1), [pa.record_batch({"k": [2, 2, 2], "v": [2, 2, 2]})])
    assert _keys(fetch(srv.addr, ShuffleTicket(1, 0, 0, 0))) == [1, 1]
    assert _keys(fetch(srv.addr, ShuffleTicket(1, 0, 0, 1))) == [2, 2, 2]


def test_explicit_credit_window_bounds_producer_inflight():
    """A Carbonite-granted window bounds how far the producer runs ahead.

    With a small window the producer must never buffer more than `window` batches
    ahead of the reducer (verified via the in-flight high-water mark), yet every
    batch still arrives — the flow-control guarantee that a fast producer can't OOM
    a slow consumer."""
    srv = FlightShuffleServer()
    ticket = ShuffleTicket(7, 0, 0, 0)
    batches = [pa.record_batch({"k": [i] * 100, "v": [i] * 100}) for i in range(30)]
    srv.publish(ticket, batches)

    window = 2
    got = fetch(srv.addr, ticket, credits=window)
    assert sum(b.num_rows for b in got) == sum(b.num_rows for b in batches)

    high_water = srv.max_inflight(ticket)
    assert high_water is not None and 1 <= high_water <= window, (
        f"producer ran {high_water} batches ahead, exceeding the granted window {window}"
    )
