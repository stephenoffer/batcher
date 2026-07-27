"""A finished query's shuffle buckets must actually be freed.

The Flight `PartitionStore` is append-only: a published bucket stays resident until
something evicts it. Every eviction call existed and was bound end to end —
`ShuffleSession.{release, clear_plan, clear}` through `bc_py` to the Rust store, with unit
tests on the Rust side — and on the batch path **nothing called any of them**. Only
`dist/streaming/pipeline.py` released anything.

With `distributed.reuse_session_fleet` on by default the worker fleet outlives the query
that made it, so a session accumulated every bucket of every stage of every query until
the node ran out of memory. The symptom is an out-of-memory kill on the Nth query, with
nothing pointing at queries 1..N-1.

A leak has no symptom until it is fatal, so these tests assert on the one thing that can
see it directly: what a session is still holding, via `ShuffleSession.partition_count`.
"""

from __future__ import annotations

import pytest

from batcher.carbonite.transfer import ShuffleSession, ShuffleTicket

pytestmark = pytest.mark.integration


def _publish(session: ShuffleSession, plan: int, stage: int, src: int, dst: int) -> ShuffleTicket:
    """Publish one non-empty bucket and return its ticket."""
    import pyarrow as pa

    ticket = ShuffleTicket(plan, stage, src, dst, 0)
    batch = pa.record_batch({"a": pa.array([1, 2, 3], type=pa.int64())})
    session.publish(ticket, [batch])
    return ticket


class TestPlanScopedEviction:
    def test_clear_plan_frees_only_that_plan(self) -> None:
        """The property a session fleet depends on: one query's teardown must not evict
        another concurrent query's live buckets, and must evict all of its own."""
        session = ShuffleSession()
        try:
            for dst in range(3):
                _publish(session, plan=1, stage=0, src=0, dst=dst)
            for dst in range(2):
                _publish(session, plan=2, stage=0, src=0, dst=dst)
            assert session.partition_count == 5

            session.clear_plan(1)

            # Plan 1 gone, plan 2 untouched.
            assert session.partition_count == 2
            assert session.gather([(session.addr, ShuffleTicket(1, 0, 0, 0, 0))]) == []
            assert session.gather([(session.addr, ShuffleTicket(2, 0, 0, 0, 0))]) != []
        finally:
            session.clear()

    def test_a_multi_stage_query_frees_every_stage(self) -> None:
        """All of one query's stages share a plan id, so teardown must sweep the lot.

        This is the cross-query half of the leak: a staged query publishes under one plan
        id at several stages, and evicting only the last stage would leave the rest.
        """
        session = ShuffleSession()
        try:
            for stage in range(4):
                _publish(session, plan=7, stage=stage, src=0, dst=0)
            assert session.partition_count == 4
            session.clear_plan(7)
            assert session.partition_count == 0
        finally:
            session.clear()

    def test_evicting_an_unknown_plan_is_harmless(self) -> None:
        # Teardown runs in a `finally` on every query, including ones that published
        # nothing. It must never raise there.
        session = ShuffleSession()
        try:
            _publish(session, plan=1, stage=0, src=0, dst=0)
            session.clear_plan(999)
            assert session.partition_count == 1
        finally:
            session.clear()


class TestTicketScopedRelease:
    def test_release_frees_one_bucket(self) -> None:
        """The per-stage release, which is what bounds memory *within* one query.

        Without it, stage `k`'s buckets stay resident through stages `k+1..n`, so a deep
        adaptive query holds every intermediate at once — a leak that kills a single query
        rather than a session.
        """
        session = ShuffleSession()
        try:
            tickets = [_publish(session, plan=3, stage=0, src=0, dst=d) for d in range(3)]
            assert session.partition_count == 3

            session.release(tickets[0])
            assert session.partition_count == 2
            # The released bucket reads back EMPTY rather than raising — the epoch
            # invariant. That is exactly why eviction is only ever done at points where
            # everything downstream has finished: a premature release loses rows silently.
            assert session.gather([(session.addr, tickets[0])]) == []
            assert session.gather([(session.addr, tickets[1])]) != []
        finally:
            session.clear()


def test_repeated_queries_on_one_session_do_not_accumulate() -> None:
    """The leak itself, in miniature: N queries through one long-lived session.

    Before eviction was wired, `partition_count` grew monotonically here and never came
    back down — which is precisely what exhausted a warm session fleet's memory in
    production. It must return to zero after each query's teardown.
    """
    session = ShuffleSession()
    try:
        for plan in range(1, 6):
            for dst in range(4):
                _publish(session, plan=plan, stage=0, src=0, dst=dst)
            assert session.partition_count == 4, "buckets from an earlier query survived"
            session.clear_plan(plan)
            assert session.partition_count == 0, f"plan {plan} leaked buckets"
    finally:
        session.clear()
