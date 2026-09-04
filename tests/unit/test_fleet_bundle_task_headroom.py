"""A fleet bundle keeps a sliver its actor does not claim, so the query's tasks can run.

A shuffle fleet is a placement-group reservation of one worker per node holding that node's
cores — i.e. **100% of the cluster's schedulable CPU**. A stage of the same query that runs
as plain Ray tasks submits them outside that reservation, so while the fleet is up they are
not slow, they are unplaceable. Measured on a 4x96 cluster as
`{'CPU': 0.125, 'memory': 1048576.0}: 1+ pending` against `384.0/384.0`, indefinitely, from a
three-table join whose final stage scans the intermediate the fleet is holding
(`tests/integration/test_distributed.py::test_distributed_multi_table_join_matches_single_node[flight]`).

`yield_session_fleet` handles the releasable case. This handles the one it cannot: an
intermediate published on the actors, where a teardown would be a wrong answer rather than a
slow one. The bundle keeps `fleet_task_headroom` free and the stage runs inside it.

The memory half is the part that is easy to get wrong and was: a bundle's `memory` is a
scheduling *hint* sized from the query's envelope, and it is routinely smaller than a single
task's ask — 1 MiB per bundle against a task asking for exactly 1 MiB — so no headroom inside
it could ever fit, and a CPU-only sliver left the task pending on memory instead of on cores.
"""

from __future__ import annotations

import pytest

from batcher.dist.executors.ray_runtime import scheduling

pytestmark = pytest.mark.unit


def test_headroom_is_a_percent_of_a_node_sized_grant_and_never_takes_the_actor_below_a_core():
    """One core out of 96 bounds the hang; a one-core worker has nothing to spare."""
    assert scheduling.fleet_task_headroom(96.0) == 1.0
    assert scheduling.fleet_task_headroom(48.0) == 1.0
    assert scheduling.fleet_task_headroom(8.0) == 1.0
    assert scheduling.fleet_task_headroom(4.0) == 0.5
    assert scheduling.fleet_task_headroom(1.0) == 0.0
    assert scheduling.fleet_task_headroom(0.5) == 0.0


def test_an_actor_claims_less_than_its_bundle_reserves(monkeypatch):
    """The bundle is sized from the envelope; the actor takes that minus the sliver."""
    grant = {"num_cpus": 96.0, "memory": 1 << 20}
    monkeypatch.setattr(scheduling, "task_options", lambda _env: dict(grant))
    monkeypatch.setattr(scheduling, "current_envelope", lambda: None)
    monkeypatch.setattr(scheduling, "placement_actor_options", lambda pg, i, base: dict(base))

    opts = scheduling.fleet_actor_options(object(), 2)
    assert [o["num_cpus"] for o in opts] == [95.0, 95.0]
    # Memory is untouched: see the module docstring — the fix for it is on the task side.
    assert {o["memory"] for o in opts} == {1 << 20}


def test_without_a_placement_group_the_grant_is_untouched(monkeypatch):
    """No bundle means no reservation to leave room in, so a lone actor keeps its whole grant."""
    monkeypatch.setattr(scheduling, "task_options", lambda _env: {"num_cpus": 96.0})
    monkeypatch.setattr(scheduling, "current_envelope", lambda: None)
    monkeypatch.setattr(scheduling, "placement_actor_options", lambda pg, i, base: dict(base))

    assert scheduling.fleet_actor_options(None, 1)[0]["num_cpus"] == 96.0


def test_fleet_task_options_drops_the_memory_hint():
    """A bundle's memory hint can be smaller than one task's ask, so the ask has to go."""
    assert scheduling.fleet_task_options(None) == {}
    opts = scheduling.fleet_task_options(_FakePg())
    assert opts["memory"] == 0
    assert opts["scheduling_strategy"].placement_group is not None


class _FakePg:
    """A stand-in accepted by `PlacementGroupSchedulingStrategy` without a live cluster."""

    id = b"\x00" * 18
    bundle_specs = ({"CPU": 1.0},)
