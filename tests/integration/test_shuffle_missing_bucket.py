"""A shuffle bucket that should exist and does not is lost data, never an empty bucket.

Every shuffle mapper publishes every bucket, empty ones included, so a reducer can know which
tickets must exist. The transport used to read an absent ticket as an empty partition, which
turned a bucket evicted early, lost with its spill file, or never copied to a replica into a
successful query with fewer rows. These tests remove a bucket after it was published and
check each reader path reports the source as unreachable (so the driver recomputes it) or
raises, and never returns short.

They run real Flight servers in this process and need no Ray cluster.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher._internal.errors import RetryableShuffleError
from batcher.carbonite.transfer.server import ShuffleClient
from batcher.carbonite.transfer.session import ShuffleSession, ShuffleTicket

pytestmark = pytest.mark.integration


def _bucket(v: int) -> list[pa.RecordBatch]:
    return [pa.record_batch({"k": [v, v + 1, v + 2]})]


def _three_mappers(plan: int):
    """Three mapper sessions, each publishing bucket 0 for one reducer, and that reducer."""
    mappers = [ShuffleSession() for _ in range(3)]
    tickets = [ShuffleTicket(plan, 0, src, 0) for src in range(3)]
    for src, (m, t) in enumerate(zip(mappers, tickets, strict=True)):
        m.publish(t, _bucket(src * 10))
    return mappers, tickets, ShuffleSession()


def test_a_released_bucket_is_reported_unreachable_not_empty():
    """BT-001: remove one expected ticket after publication; the gather must name it."""
    mappers, tickets, reducer = _three_mappers(9101)
    mappers[1].release(tickets[1])  # the bucket vanishes after publication

    sources = [(m.addr, t) for m, t in zip(mappers, tickets, strict=True)]
    rows, unreachable = reducer.gather_concat(sources)
    assert [idx for idx, _ in unreachable] == [1], (
        "a missing bucket must come back as an unreachable source for the driver to "
        f"recompute, got {unreachable}"
    )
    # The other two sources still arrive; the driver discards a partial round anyway.
    assert sum(b.num_rows for b in rows) == 6


def test_an_empty_published_bucket_is_still_empty_not_missing():
    """The distinction cuts both ways: a published empty bucket is a clean zero-row source."""
    mapper, reducer = ShuffleSession(), ShuffleSession()
    ticket = ShuffleTicket(9102, 0, 0, 0)
    mapper.publish(ticket, [])
    rows, unreachable = reducer.gather_concat([(mapper.addr, ticket)])
    assert unreachable == []
    assert sum(b.num_rows for b in rows) == 0


def test_a_missing_co_located_bucket_is_reported_unreachable():
    """The no-socket path: the reducer's own store lost the only copy."""
    session = ShuffleSession()
    ticket = ShuffleTicket(9103, 0, 0, 0)
    session.publish(ticket, _bucket(1))
    session.release(ticket)
    _rows, unreachable = session.gather_concat([(session.addr, ticket)])
    assert [idx for idx, _ in unreachable] == [0]


def test_a_required_fetch_raises_where_a_lenient_one_reads_empty():
    """Replication and stage-output reads use `required=True`; absence must raise there."""
    mapper = ShuffleSession()
    ticket = ShuffleTicket(9104, 0, 0, 0)  # never published
    client = ShuffleClient()
    assert client.fetch(mapper.addr, ticket) == []  # lenient: streaming polls rely on it
    with pytest.raises(RetryableShuffleError):
        client.fetch(mapper.addr, ticket, required=True)
    with pytest.raises(RetryableShuffleError):
        mapper.fetch(mapper.addr, ticket, required=True)  # the DIRECT_MEMORY branch
