"""Phase-2c: per-query shuffle plan-id fencing.

A reused (persistent-fleet) actor must not serve a prior, crashed query's stale
partitions. Each query mints a fresh plan id that every ticket carries, so a stale
partition published at the same stage/src/dst/epoch under a *different* plan id can
never be fetched. These cover the id helpers and the ticket they stamp (no Ray needed
— the helpers live outside the actor's `try: import ray` block).
"""

from __future__ import annotations

from batcher.dist.flight_worker import _ticket, new_plan_id, set_current_plan_id


def test_new_plan_id_is_distinct_and_in_range():
    ids = {new_plan_id() for _ in range(100)}
    assert len(ids) == 100  # overwhelmingly unique
    assert all(0 <= i < (1 << 63) for i in ids)


def test_ticket_carries_the_current_plan_id():
    p = new_plan_id()
    set_current_plan_id(p)
    ticket = _ticket(stage=0, src=2, dst=3, epoch=1)
    assert ticket.plan_id == p
    assert (ticket.stage_id, ticket.src_partition, ticket.dst_partition, ticket.epoch) == (
        0,
        2,
        3,
        1,
    )


def test_two_queries_get_fenced_tickets():
    set_current_plan_id(new_plan_id())
    first = _ticket(0, 1, 1)
    set_current_plan_id(new_plan_id())  # a second query on the same process/actor
    second = _ticket(0, 1, 1)
    # Same stage/src/dst/epoch, but different plan id → distinct ticket strings, so a
    # stale partition from the first query can never be fetched by the second.
    assert first.plan_id != second.plan_id
    assert str(first) != str(second)
