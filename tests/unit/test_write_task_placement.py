"""A distributed write must be schedulable against a cluster its own query has reserved.

`sort(...).write.parquet(distributed=True)` is the ordinary shape, and the two halves fight:
the sort's shuffle fleet reserves every schedulable core, and the write submits plain Ray
tasks **outside** that reservation. The fleet does not slow those tasks down, it makes them
unplaceable — and a placement that can never happen has no error and no timeout, so the
barrier waits forever reporting `0/N tasks finished, cluster CPU 384/384 in use`.

The same fragment carries the shard's CPU share, because the two are one question: the
accommodation reserves headroom for exactly these tasks, so it has to be told what one of them
costs. A shard pushes bytes at object storage and waits on the network rather than saturating a
core, which is what `execution.cpu_share_io` is for and what this path used to ignore.

Reproduced on a 4 x 96-core cluster with nothing else running, from that one statement in a
fresh process. `tests/integration/test_distributed_no_materialize.py` is the end-to-end
proof (it hung indefinitely and now runs in 45 s); these tests pin the mechanism, so a
refactor that drops the accommodation fails fast here instead of hanging a cluster suite.
"""

from __future__ import annotations

import pytest

pytest.importorskip("ray", reason="the fleet accommodation lives behind the optional ray extra")

from batcher.dist.executors.write import _write_task_cpus, _write_task_scheduling

pytestmark = pytest.mark.unit


def _sched(task_cpus: float = 0.5) -> dict:
    """The scheduling fragment with the always-present CPU share stripped back off.

    Every assertion below is about the *accommodation*, which is the half that can hang a
    query. Removing the share here keeps those assertions reading as they did rather than
    restating it four times.
    """
    opts = _write_task_scheduling(task_cpus)
    assert opts.pop("num_cpus") == task_cpus
    return opts


@pytest.fixture
def fleet(monkeypatch):
    """Stand in for `dist.fleet`, so the two branches can be driven without a cluster."""
    import batcher.dist.fleet as mod

    state = {"yielded": False, "pg": None}

    def yield_session_fleet(needed_cpus: float) -> bool:
        state["asked_for"] = needed_cpus
        return state["yielded"]

    monkeypatch.setattr(mod, "yield_session_fleet", yield_session_fleet)
    monkeypatch.setattr(mod, "held_placement_group", lambda: state["pg"])
    return state


def test_an_open_cluster_leaves_the_write_exactly_as_it_was(fleet) -> None:
    """No fleet, no accommodation: the ordinary path must not acquire a placement strategy."""
    fleet["yielded"] = False
    fleet["pg"] = None
    assert _sched() == {}


def test_a_fleet_that_can_be_handed_back_is_handed_back(fleet) -> None:
    """The preferred answer: release the reservation and let the write have the cluster.

    Cheaper than running inside the bundles, because the tasks then get whole nodes rather
    than the sliver `fleet_task_headroom` leaves — and the next shuffle stage simply
    respawns the fleet.
    """
    fleet["yielded"] = True
    fleet["pg"] = object()  # would be used if the yield had not succeeded
    assert _sched() == {}
    assert fleet["asked_for"] > 0


def test_a_fleet_holding_an_intermediate_is_run_inside_instead(fleet, monkeypatch) -> None:
    """When the fleet cannot be released, the tasks must be scheduled into its own group.

    This is the branch whose absence was the deadlock: without it the tasks are submitted
    against a cluster with no free cores and pend forever. Asserted as "the held group is
    what the write asks to be placed into" rather than by building a real
    `PlacementGroupSchedulingStrategy`, because constructing one reaches for a live cluster
    and a unit test must not.
    """
    import batcher.dist.executors.ray_runtime as rr

    seen = {}

    def fake_options(pg):
        seen["pg"] = pg
        return {"scheduling_strategy": "into-the-fleet"}

    monkeypatch.setattr(rr, "fleet_task_options", fake_options)
    sentinel = object()
    fleet["yielded"] = False
    fleet["pg"] = sentinel

    assert _sched() == {"scheduling_strategy": "into-the-fleet"}
    assert seen["pg"] is sentinel


def test_a_probe_that_raises_never_fails_the_write(fleet, monkeypatch) -> None:
    """Scheduling is a courtesy: a write must not fail because the accommodation could not.

    Without this the fix would trade a hang for a crash on any cluster whose fleet state
    cannot be read, which is a worse failure than the one it replaces.
    """
    import batcher.dist.fleet as mod

    def boom(_needed):
        raise RuntimeError("no fleet state")

    monkeypatch.setattr(mod, "yield_session_fleet", boom)
    assert _sched() == {}


# --- the shard's CPU share ------------------------------------------------------------------


def test_a_shard_asks_for_the_io_share_rather_than_a_whole_core() -> None:
    """A write shard waits on the network. Reserving a core per shard caps a 96-core cluster
    at 96 concurrent uploads that are nearly all idle."""
    from batcher.config import active_config

    assert _write_task_cpus() == active_config().execution.cpu_share_io


def test_a_shard_never_asks_for_more_than_the_fleet_was_granted() -> None:
    """The IO share is a prior, not an override. An envelope that already asked for less
    derived that from the plan, and raising it here would over-reserve the node."""
    from batcher.dist.executors.ray_runtime import (
        reset_scheduling_envelope,
        set_scheduling_envelope,
    )
    from batcher.plan.resource import SchedulingEnvelope

    token = set_scheduling_envelope(SchedulingEnvelope(num_cpus=0.3))
    try:
        assert _write_task_cpus() == pytest.approx(0.3)
    finally:
        reset_scheduling_envelope(token)


def test_a_shard_never_asks_for_an_unschedulable_sliver() -> None:
    """Floored at `cpu_share_min` for the same reason the map path floors its own share: a
    task asking for a thousandth of a core packs without limit and oversubscribes the node."""
    from batcher.config import active_config
    from batcher.dist.executors.ray_runtime import (
        reset_scheduling_envelope,
        set_scheduling_envelope,
    )
    from batcher.plan.resource import SchedulingEnvelope

    token = set_scheduling_envelope(SchedulingEnvelope(num_cpus=0.001))
    try:
        assert _write_task_cpus() == active_config().execution.cpu_share_min
    finally:
        reset_scheduling_envelope(token)
