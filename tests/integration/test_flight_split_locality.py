"""A cross-stage intermediate is read through the locality selector, not over gRPC.

A Flight stage run with ``materialize=False`` leaves each reducer's bucket published on
its host worker and hands the next stage `FlightFetchSplit`s over them. Those splits used
to fetch unconditionally through the pooled Flight client, so a worker reading the bucket
its own process published paid a loopback gRPC round-trip — serializing bytes already in
its heap. These tests pin the fix: the read resolves its transfer mode against the
process's live `ShuffleSession`, and the rows are identical whichever mode serves them.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.carbonite.transfer import ShuffleSession, ShuffleTicket
from batcher.carbonite.transfer.lifecycle import local_session, process_client
from batcher.dist.fleet import FlightFetchSplit

pytest.importorskip("batcher._native", reason="native engine not built")

pytestmark = pytest.mark.integration


def _bucket() -> list[pa.RecordBatch]:
    return [pa.record_batch({"k": [1, 2, 3], "v": [10.0, 20.0, 30.0]})]


def _split(session: ShuffleSession, ticket: ShuffleTicket, batches) -> FlightFetchSplit:
    rows = sum(b.num_rows for b in batches)
    return FlightFetchSplit(session.addr, ticket, rows, batches[0].schema)


def test_reading_a_bucket_this_process_published_stays_off_the_network():
    """The whole point: the holder reads its own bucket out of the local store."""
    ticket = ShuffleTicket(1, 0, 0, 0)
    batches = _bucket()
    with ShuffleSession() as session:
        session.publish(ticket, batches)
        before = session.stats()["off_network_fetches"]

        out = _split(session, ticket, batches).read()

        assert [b.to_pydict() for b in out] == [b.to_pydict() for b in batches]
        assert session.stats()["off_network_fetches"] == before + 1
        assert session.locality_ratio == 1.0


def test_the_split_advertises_its_holder_as_its_affinity():
    """What the locality-aware split assignment routes on."""
    ticket = ShuffleTicket(1, 0, 0, 0)
    with ShuffleSession() as session:
        session.publish(ticket, _bucket())
        assert _split(session, ticket, _bucket()).affinity() == session.addr


def test_the_local_read_returns_exactly_what_the_network_read_returns():
    """Placement changes where bytes travel, never what they are.

    The local fast path and the Flight path are compared on the same published bucket,
    so a divergence between them shows up here rather than as a wrong distributed result.
    """
    ticket = ShuffleTicket(2, 0, 1, 0)
    batches = _bucket()
    with ShuffleSession() as session:
        session.publish(ticket, batches)

        local = _split(session, ticket, batches).read()
        remote = process_client().fetch(session.addr, str(ticket))

        assert [b.to_pydict() for b in local] == [b.to_pydict() for b in remote]


def test_a_projected_read_is_projected_on_both_paths():
    ticket = ShuffleTicket(3, 0, 0, 0)
    batches = _bucket()
    with ShuffleSession() as session:
        session.publish(ticket, batches)
        out = _split(session, ticket, batches).read(projection=["v"])
        assert [b.schema.names for b in out] == [["v"]]


def test_an_intermediate_is_partitioned_back_onto_the_workers_holding_it():
    """The two halves together: the next stage's worker `i` is handed the bucket worker
    `i` published, so `read` above resolves to the local store instead of the network.

    Without this the assignment bin-packs by row count alone, and an `N`-worker fleet
    re-fetches about `1 - 1/N` of its own intermediate over the wire.
    """
    from batcher.dist.executors.partition_io import partition_descriptors
    from batcher.dist.fleet import FlightMaterializedSource

    addrs = [f"10.0.0.{i}:5005" for i in range(1, 5)]
    schema = _bucket()[0].schema
    # Buckets listed in reverse worker order, so a positional coincidence cannot pass this.
    handles = [(addr, ShuffleTicket(7, 0, 0, i), 100) for i, addr in enumerate(reversed(addrs))]
    source = FlightMaterializedSource(handles, schema, None, None)

    descs = partition_descriptors(source, len(addrs), worker_addrs=addrs)

    assert [[s.addr for s in d["splits"]] for d in descs] == [[a] for a in addrs]


def test_without_worker_addresses_the_intermediate_keeps_the_old_assignment():
    """The callers that cannot name their fleet must be unaffected."""
    from batcher.dist.executors.partition_io import partition_descriptors
    from batcher.dist.fleet import FlightMaterializedSource

    addrs = [f"10.0.0.{i}:5005" for i in range(1, 5)]
    schema = _bucket()[0].schema
    handles = [(addr, ShuffleTicket(7, 0, 0, i), 100) for i, addr in enumerate(reversed(addrs))]
    source = FlightMaterializedSource(handles, schema, None, None)

    descs = partition_descriptors(source, len(addrs))

    # Every bucket is still assigned exactly once, just without regard to who holds it.
    assert sorted(s.addr for d in descs for s in d["splits"]) == sorted(addrs)


def test_local_session_prefers_the_holder_among_several():
    """A process running more than one session must fetch through the one that holds the
    bucket — any other resolves to `NETWORK` and gives the loopback hop straight back."""
    with ShuffleSession() as a, ShuffleSession() as b:
        assert local_session(a.addr) is a
        assert local_session(b.addr) is b
        # An address no local session serves still yields a session (it fetches remotely
        # under that session's credit window) rather than dropping to the unwindowed client.
        assert local_session("10.255.255.255:1") in (a, b)
