"""Cluster-correctness of the Carbonite Flight transport (N1, C1/N2).

These guard the two latent multi-node bugs that a single-host local Ray cluster
never exercised: a loopback-only advertised address (unreachable cross-node), and a
`gather` that swallowed every fetch error into an empty bucket (silently wrong).
"""

from __future__ import annotations

import pytest

from batcher.carbonite.transfer import ShuffleSession, ShuffleTicket
from batcher.carbonite.transfer.server import FlightShuffleServer

pytestmark = pytest.mark.integration


def test_advertise_host_is_routable_not_loopback():
    # N1: with an advertise host set, the server advertises {host}:{port} — a
    # routable address a reducer on another node can dial — not 127.0.0.1.
    srv = FlightShuffleServer(advertise_host="10.1.2.3")
    host, _, port = srv.addr.rpartition(":")
    assert host == "10.1.2.3"
    assert port.isdigit() and int(port) > 0


def test_default_advertise_stays_loopback():
    # No advertise host → single-host loopback behavior is unchanged.
    srv = FlightShuffleServer()
    assert srv.addr.startswith("127.0.0.1:")


def test_gather_unpublished_ticket_is_empty_not_error():
    # C1/N2: a mapper that produced no rows never publishes the ticket; fetching it
    # is the expected empty-bucket case and must yield an empty result (NotFound is
    # mapped to empty at the transport boundary), not raise.
    a = ShuffleSession()
    b = ShuffleSession()
    missing = ShuffleTicket(1, 0, 0, 0, 0)
    assert b.gather([(a.addr, missing)]) == []


def test_gather_unreachable_peer_raises_not_silent_empty():
    # C1/N2: a *real* fault (an unreachable peer) must propagate — silently
    # treating it as an empty bucket would return wrong results. Port 1 is in the
    # privileged range and not serving Flight, so the connect fails.
    b = ShuffleSession()
    ticket = ShuffleTicket(1, 0, 0, 0, 0)
    with pytest.raises(Exception):  # noqa: B017 - any transport error, just not silent
        b.gather([("127.0.0.1:1", ticket)])


def _batch():
    import pyarrow as pa

    return pa.record_batch({"k": [1, 2], "v": [10, 20]})


def test_shuffle_auth_token_required_when_configured():
    # N5: an auth-gated server serves a fetch that presents the matching token, and
    # rejects one that presents a wrong/absent token — so a process that merely
    # reaches the port cannot exfiltrate shuffle partitions.
    server = ShuffleSession(token="s3cret")
    ticket = ShuffleTicket(1, 0, 0, 0, 0)
    server.publish(ticket, [_batch()])

    # A matching token from a different (network) session succeeds.
    good = ShuffleSession(token="s3cret")
    assert sum(b.num_rows for b in good.fetch(server.addr, ticket)) == 2

    # A wrong token is rejected (Unauthenticated → a raised error, not empty).
    bad = ShuffleSession(token="wrong")
    with pytest.raises(Exception):  # noqa: B017 - any auth error, just not silent success
        bad.fetch(server.addr, ticket)

    # No token at all is also rejected.
    anon = ShuffleSession()
    with pytest.raises(Exception):  # noqa: B017
        anon.fetch(server.addr, ticket)


def test_store_eviction_bounds_retained_partitions():
    # C8/C9: the partition store is not append-only — release/clear_plan/clear free
    # published partitions so a long-lived worker doesn't accumulate every stage.
    s = ShuffleSession()
    for stage in range(5):
        s.publish(ShuffleTicket(1, stage, 0, 0, 0), [_batch()])
    assert s.partition_count == 5

    s.release(ShuffleTicket(1, 0, 0, 0, 0))
    assert s.partition_count == 4

    s.publish(ShuffleTicket(2, 0, 0, 0, 0), [_batch()])  # a different plan
    s.clear_plan(1)  # evict only plan 1's remaining stages
    assert s.partition_count == 1  # plan 2's partition survives

    s.clear()
    assert s.partition_count == 0
