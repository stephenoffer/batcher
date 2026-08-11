"""Distributed placement diagnostics: why a Ray request is pending, and who says so.

`ray.wait` cannot distinguish a task queued behind a busy cluster from one asking for more
than any node has, and the second never finishes. Neither could the engine, so every stalled
barrier, failed fleet spawn, and abandoned placement group reported the same unresolved
"go run `ray status`". These tests pin what each of them says now:

* `describe_pending_demand`'s three answers — unsatisfiable, short, and nothing-wrong —
  against a stubbed topology, so the boundaries are asserted rather than assumed;
* `workers_per_node`, the single per-node placement rule every one of those questions
  reduces to (it was stated three times and drifted three ways);
* the topology snapshot covering the per-node free-CPU read it used to leave live;
* the fleet-spawn failure naming the reason, and a fleet that comes up narrow saying so;
* the once-per-session report of which Ray this process actually attached to.

No cluster is needed: the topology is stubbed at `scaling.node_classes`, the one function
every one of these reads it through.
"""

from __future__ import annotations

import pytest

from batcher.dist.executors.ray_runtime import capacity
from batcher.dist.executors.ray_runtime.capacity import Demand, describe_pending_demand
from batcher.plan.resource import SchedulingEnvelope

pytestmark = pytest.mark.unit


def _node(cpus: float, *, free: float | None = None, gpus: float = 0.0, memory: float = 0.0):
    return {
        "node_id": f"n{cpus}-{free}-{gpus}",
        "cpus": cpus,
        "free_cpus": cpus if free is None else free,
        "gpus": gpus,
        "memory": memory,
        "accelerators": 0.0,
        "accelerator_type": None,
    }


@pytest.fixture
def topology(monkeypatch):
    """Install a fake cluster topology and cluster-resource totals.

    Both are patched on the module the diagnosis imports them *into* usage at call time, so
    the stub is what the function under test actually reads.
    """

    def _install(nodes, totals=None):
        import batcher.dist.executors.ray_runtime.scaling as scaling

        monkeypatch.setattr(scaling, "node_classes", lambda: list(nodes))
        monkeypatch.setattr(
            capacity, "_missing_custom_resources", lambda d: _missing(d, totals or {})
        )

    def _missing(demand, totals):
        return [name for name, amount in demand.resources if totals.get(name, 0.0) < amount]

    return _install


def test_idle_cluster_that_fits_the_ask_has_no_complaint(topology):
    """A demand every node can host, on a cluster with cores free, is not a diagnosis.

    The `None` case is the one that matters most: a diagnosis emitted for an ordinary slow
    task trains a reader to skip the ones that mean something.
    """
    topology([_node(16), _node(16)])
    assert describe_pending_demand(Demand(num_cpus=4, count=8)) is None


def test_ask_wider_than_every_node_is_reported_as_unsatisfiable(topology):
    """A per-task CPU grant no single node can host can never be placed.

    This is the shape that hangs a query forever: Ray keeps the task pending, the barrier
    keeps waiting, and nothing times out. The message must say so rather than suggest
    waiting.
    """
    topology([_node(8), _node(16)])
    msg = describe_pending_demand(Demand(num_cpus=32))
    assert msg is not None
    assert "no node can host one task" in msg
    assert "CPU (16 available, 32 needed)" in msg


def test_the_binding_resource_is_named_not_just_the_cpu_count(topology):
    """A memory-bound ask names memory, not the cores that were never the problem.

    Reporting the widest node's *cores* against a memory shortfall reads as an engine bug:
    the reader compares 16 CPU to the 1 CPU asked for and concludes the diagnosis is wrong.
    """
    topology([_node(16, memory=32e9)])
    msg = describe_pending_demand(Demand(num_cpus=1, memory_bytes=int(64e9)))
    assert msg is not None
    assert "short on memory (32.0 GB available, 64.0 GB needed)" in msg


def test_a_node_that_does_not_report_memory_is_not_treated_as_having_none(topology):
    """Ray not tracking `memory` on a node must not rule that node out.

    Reading an absent figure as zero would declare every such cluster unsatisfiable for any
    memory grant at all — which is the common case, since `memory` is an optional resource.
    """
    topology([_node(16, memory=0.0)])
    assert describe_pending_demand(Demand(num_cpus=1, memory_bytes=int(64e9))) is None


def test_gpu_ask_on_a_cpu_only_cluster_is_unsatisfiable(topology):
    topology([_node(16), _node(16)])
    msg = describe_pending_demand(Demand(num_cpus=1, num_gpus=2))
    assert msg is not None
    assert "short on GPU (0 available, 2 needed)" in msg


def test_absent_custom_resource_is_named_before_the_per_node_test(topology):
    """A `TPU`/`neuron_cores` ask the cluster has none of is its own answer.

    Custom resources are checked against the cluster total, so the message must name the
    resource rather than fall through to a CPU comparison that says nothing about it.
    """
    topology([_node(16)], totals={"CPU": 32.0})
    msg = describe_pending_demand(Demand(num_cpus=1, resources=(("TPU", 4.0),)))
    assert msg is not None
    assert "no node advertises TPU" in msg


def test_a_full_cluster_is_reported_as_short_not_impossible(topology):
    """Nodes that could host the ask but hold no free cores is a wait, not a failure.

    The distinction is the whole point: this one resolves on its own once the co-tenant
    finishes, so the message must not tell the reader the query is doomed.
    """
    topology([_node(16, free=0.0), _node(16, free=0.0)])
    msg = describe_pending_demand(Demand(num_cpus=8, count=4))
    assert msg is not None
    assert "short of free capacity" in msg
    assert "no node can host" not in msg


def test_a_cluster_holding_only_a_fraction_of_the_outstanding_demand_says_so(topology):
    """The real shape of a stalled shared cluster: room for one task, not for the stage.

    Bounding on "can even one task fit" reported nothing on a 128-core cluster with 8 cores
    free and sixteen 8-core tasks outstanding — which is a stage that will make essentially no
    progress, and the exact case the warning exists for. The comparison is against what is
    outstanding, not against a single task.
    """
    topology([_node(16, free=4.0), _node(16, free=4.0)])
    msg = describe_pending_demand(Demand(num_cpus=8, count=16))
    assert msg is not None
    assert "8 CPU free between them" in msg
    assert "16 outstanding" in msg


def test_room_for_everything_outstanding_is_not_a_diagnosis(topology):
    """A cluster that can start the whole stage now has nothing wrong with it."""
    topology([_node(16), _node(16)])
    assert describe_pending_demand(Demand(num_cpus=2, count=8)) is None


def test_an_unreadable_topology_produces_no_diagnosis(topology):
    """A cluster whose topology cannot be read still has to run the query.

    `node_classes` returns `[]` on any failure, and guessing from nothing would be worse
    than saying nothing.
    """
    topology([])
    assert describe_pending_demand(Demand(num_cpus=1000)) is None


def test_demand_from_envelope_carries_every_reserved_axis():
    """The diagnosis must be made against the same ask the bundle reserved.

    `scheduling._bundle` reserves CPU, GPU, memory and custom resources; a `Demand` built
    from the envelope that dropped any of them would clear a request Ray had refused.
    """
    env = SchedulingEnvelope(
        num_cpus=4.0,
        memory_bytes=1024,
        num_gpus=2.0,
        n_tasks=7,
        resources=(("TPU", 8.0),),
    )
    demand = Demand.from_envelope(env)
    assert demand.num_cpus == 4.0
    assert demand.num_gpus == 2.0
    assert demand.memory_bytes == 1024
    assert demand.resources == (("TPU", 8.0),)
    assert demand.count == 7


def test_demand_from_no_envelope_is_rays_implicit_one_cpu():
    """With no Carbonite grant a task holds Ray's default, and the diagnosis must match it."""
    demand = Demand.from_envelope(None, count=3)
    assert demand.num_cpus == 1.0
    assert demand.num_gpus == 0.0
    assert demand.count == 3


# --- the topology snapshot ----------------------------------------------------------------


def test_the_topology_snapshot_covers_the_per_node_free_cpu_read(monkeypatch):
    """Inside a scope, `node_classes` must make no GCS round trip at all.

    The snapshot collapsed the `ray.nodes()` reads and left this one live, so a placement
    phase that calls `node_classes` from five places (the pack decision, the node-class
    selector, `placeable_workers`, the zone selector, this diagnosis) still paid five round
    trips for the free-CPU half of the same question. Measured at 10.6 ms for five calls
    against 0.1 ms once the snapshot covers it.
    """
    from batcher.dist.executors.ray_runtime import capacity as cap
    from batcher.dist.executors.ray_runtime import scaling

    reads: list[int] = []

    def _counted():
        reads.append(1)
        return {"node-a": 4.0}

    monkeypatch.setattr(cap, "_live_free_cpus_by_node", _counted)
    monkeypatch.setattr(scaling, "_read_draining", frozenset)
    monkeypatch.setattr(
        scaling,
        "_read_topology",
        lambda: scaling._Topology([], {}, frozenset(), _counted()),
    )

    with scaling.topology_scope():
        for _ in range(5):
            assert cap.free_cpus_by_node() == {"node-a": 4.0}
    assert len(reads) == 1, f"one read per scope, not per call (made {len(reads)})"


def test_outside_a_scope_the_free_cpu_read_stays_live(monkeypatch):
    """A caller with no scope must still see the cluster as it is right now.

    The snapshot is an optimization for one scheduling phase, not a process-wide cache: a
    stale free-CPU figure would size a fleet against capacity somebody else has taken.
    """
    from batcher.dist.executors.ray_runtime import capacity as cap

    reads: list[int] = []
    monkeypatch.setattr(cap, "_live_free_cpus_by_node", lambda: (reads.append(1), {"n": 1.0})[1])
    for _ in range(3):
        cap.free_cpus_by_node()
    assert len(reads) == 3


# --- the per-node placement rule ----------------------------------------------------------


def test_the_per_node_rule_is_one_function_for_every_caller(topology):
    """Fan-out sizing, zone choice, and the diagnosis must agree on what a node can host.

    The rule was stated three times and drifted three ways. A per-node rule that disagrees
    with itself produces a fan-out no arrangement of nodes can satisfy — which hangs the job
    at an unsatisfiable placement group rather than failing it.
    """
    from batcher.dist.executors.ray_runtime.capacity import placeable_workers, workers_per_node

    node = _node(16, free=4.0, gpus=2.0, memory=64e9)
    demand = Demand(num_cpus=4, num_gpus=1.0, memory_bytes=int(8e9))
    assert workers_per_node(node, demand, nameplate=True) == 2  # 4 by CPU, 2 by GPU
    assert workers_per_node(node, demand) == 1  # only 4 CPU free
    topology([node])
    assert placeable_workers(4, 1.0, memory_bytes=int(8e9)) == 2  # sizing uses nameplate


def test_fan_out_sizing_uses_nameplate_not_free_capacity(topology):
    """A fan-out is chosen before the fleet is placed, so a momentary co-tenant must not shrink it.

    Sizing off free capacity would make the same query run at a different width depending on
    who happened to be busy in the second the driver looked.
    """
    from batcher.dist.executors.ray_runtime.capacity import placeable_workers

    topology([_node(16, free=0.0), _node(16, free=0.0)])
    assert placeable_workers(4) == 8


# --- the fleet-spawn failure message ------------------------------------------------------


def test_an_unplaceable_fleet_says_why_not_just_that_it_failed(topology, monkeypatch):
    """ "over-subscribed or unschedulable" names both possibilities and distinguishes neither.

    The two have opposite fixes — change the grant, or wait for whoever holds the cluster —
    and the topology already knows which one it is. This is the error a user actually sees
    when a distributed query cannot start, so it is the one worth making actionable.
    """
    from batcher.dist.fleet import _fleet
    from batcher.plan.resource import SchedulingEnvelope

    topology([_node(8)])
    monkeypatch.setattr(
        "batcher.dist.executors.ray_runtime.scheduling.current_envelope",
        lambda: SchedulingEnvelope(num_cpus=32.0, n_tasks=4),
    )
    reason = _fleet._fleet_demand_reason()
    assert reason is not None
    assert "no node can host one task" in reason
    assert "CPU (8 available, 32 needed)" in reason


def test_a_fleet_that_comes_up_narrow_is_reported(topology, monkeypatch, caplog):
    """The most expensive silent degradation on the distributed path.

    Stragglers are killed, survivors serve the query, and on the session-cached path the rest
    of the session too — measured at 0.6 s becoming 16 s for an 8-worker join that came up
    with 2. A query running at a quarter of its width must not have to be inferred from a
    stopwatch.
    """
    import logging

    from batcher.dist.fleet import _fleet

    topology([_node(8, free=0.0)])
    with caplog.at_level(logging.WARNING, logger="batcher.dist"):
        _fleet._warn_degraded_fleet(placed=2, wanted=8, timeout=60.0)
    assert any("narrower than requested" in r.message for r in caplog.records)


def test_a_full_width_fleet_says_nothing(topology, caplog):
    """The normal case must stay silent, or the warning stops meaning anything."""
    import logging

    from batcher.dist.fleet import _fleet

    topology([_node(8)])
    with caplog.at_level(logging.WARNING, logger="batcher.dist"):
        _fleet._warn_degraded_fleet(placed=8, wanted=8, timeout=60.0)
    assert not [r for r in caplog.records if "narrower" in r.message]


# --- the attachment report ----------------------------------------------------------------


def test_the_ray_attachment_is_reported_once_per_session(monkeypatch, caplog):
    """ "Did it use the cluster, or start a local Ray?" has no error attached to it.

    A bare `ray.init()` on a workspace exporting no `RAY_ADDRESS` starts a single-node local
    Ray, and the job then runs, returns the right answer, and uses one machine's worth of
    capacity that was billed for a fleet. From inside, a one-node cluster is an ordinary
    cluster — so the engine has to say which one it attached to.
    """
    import logging

    from batcher.dist.executors.ray_runtime import lifecycle

    monkeypatch.setattr(lifecycle, "_reported_session", "")
    monkeypatch.setattr(lifecycle, "ray_session_key", lambda: "session-1")
    monkeypatch.setattr(
        lifecycle, "cluster_topology", lambda: {"nodes": 8, "cpus": 128.0, "gpus": 0.0}
    )
    monkeypatch.setattr(lifecycle, "job_ships_batcher", lambda: True)

    class _Ctx:
        gcs_address = "10.0.0.1:6379"

    class _Ray:
        @staticmethod
        def get_runtime_context():
            return _Ctx()

    with caplog.at_level(logging.INFO, logger="batcher.dist"):
        lifecycle._report_attachment(_Ray())
        lifecycle._report_attachment(_Ray())
    said = [r for r in caplog.records if "attached to Ray" in r.message]
    assert len(said) == 1, "one report per session, not one per query"


def test_a_reconnect_to_a_different_session_is_reported_again(monkeypatch, caplog):
    """A cluster restart or a notebook switching addresses is a new fact, not a repeat."""
    import logging

    from batcher.dist.executors.ray_runtime import lifecycle

    sessions = iter(["a", "a", "b"])
    monkeypatch.setattr(lifecycle, "_reported_session", "")
    monkeypatch.setattr(lifecycle, "ray_session_key", lambda: next(sessions))
    monkeypatch.setattr(
        lifecycle, "cluster_topology", lambda: {"nodes": 1, "cpus": 8.0, "gpus": 0.0}
    )
    monkeypatch.setattr(lifecycle, "job_ships_batcher", lambda: False)

    class _Ray:
        @staticmethod
        def get_runtime_context():
            return object()

    with caplog.at_level(logging.INFO, logger="batcher.dist"):
        for _ in range(3):
            lifecycle._report_attachment(_Ray())
    assert len([r for r in caplog.records if "attached to Ray" in r.message]) == 2


def test_an_unreadable_session_key_still_reports_only_once(monkeypatch, caplog):
    """`ray_session_key` returns `None` when it cannot read the session.

    Seeding the guard with `None` would make that case compare equal to the seed and never
    report; testing `is not None` first would report on every single call. `""` is the one
    seed that is not a value the key can take, so both directions stay correct.
    """
    import logging

    from batcher.dist.executors.ray_runtime import lifecycle

    monkeypatch.setattr(lifecycle, "_reported_session", "")
    monkeypatch.setattr(lifecycle, "ray_session_key", lambda: None)
    monkeypatch.setattr(
        lifecycle, "cluster_topology", lambda: {"nodes": 1, "cpus": 4.0, "gpus": 0.0}
    )
    monkeypatch.setattr(lifecycle, "job_ships_batcher", lambda: True)

    class _Ray:
        @staticmethod
        def get_runtime_context():
            return object()

    with caplog.at_level(logging.INFO, logger="batcher.dist"):
        for _ in range(3):
            lifecycle._report_attachment(_Ray())
    assert len([r for r in caplog.records if "attached to Ray" in r.message]) == 1
