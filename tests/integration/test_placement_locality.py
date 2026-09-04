"""Placement-group gang scheduling + SPREAD across a (simulated) multi-node cluster.

The Flight shuffle fleet is launched in one placement group so the whole fleet is
reserved before the shuffle starts (no partial-fleet deadlock) and SPREAD across
nodes for even data distribution and locality. These tests stand up a 2-node Ray
cluster in-process and assert the fleet actually spans both nodes — and that a
distributed query over it still equals single-node.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, count

ray = pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("batcher._native", reason="native engine not built")


@pytest.fixture(scope="module")
def _two_node_cluster():
    """Two nodes to spread a fleet across: the running cluster if it has them, else a fake one.

    The module used to build a `ray.cluster_utils.Cluster` unconditionally. That is the right
    default on a laptop and the wrong one wherever a real multi-node cluster is already up --
    and on a managed image it is worse than wrong. `Cluster(head_node_args={"num_cpus": 2})`
    there produces a session whose `cluster_resources()` carries **no `CPU` key at all**: the
    placement group is never satisfiable, `spawn_flight_workers` gives it up after its budget
    and falls back to unplaced actors, and `ray.get` on the first of them never returns. Not a
    failure -- a hang, which takes the whole integration run with it.

    So: prefer the session's own cluster when it can already spread a fleet, fall back to the
    simulated one, and skip rather than hang when neither can schedule a CPU.
    """
    import os

    def _schedulable_nodes() -> int:
        """Live nodes with a CPU to give. Two of these is what SPREAD needs to mean anything."""
        return sum(
            1 for n in ray.nodes() if n.get("Alive") and n.get("Resources", {}).get("CPU", 0) >= 1
        )

    if ray.is_initialized() or os.environ.get("RAY_ADDRESS"):
        from _ray_cluster import init_test_ray, shutdown_test_ray

        started = init_test_ray(4)
        if _schedulable_nodes() >= 2:
            yield None
            shutdown_test_ray(started)
            return
        shutdown_test_ray(started)

    from ray.cluster_utils import Cluster

    prior_address = os.environ.get("RAY_ADDRESS")
    # A prior test module may have left a single-node Ray session up; with
    # ignore_reinit_error it would silently keep that session and the fleet couldn't
    # spread. Shut it down first so this module connects to its own 2-node cluster.
    if ray.is_initialized():
        ray.shutdown()
    cluster = Cluster(
        initialize_head=True, head_node_args={"num_cpus": 2, "include_dashboard": False}
    )
    cluster.add_node(num_cpus=2)
    ray.init(address=cluster.address, logging_level="ERROR", ignore_reinit_error=True)

    def _restore() -> None:
        # Tear the private cluster down *and leave the process able to find the shared one
        # again*. `ray.init` records its address in the environment, so a module that
        # replaces the session's cluster with a 2-node, 4-CPU one and then only shuts it
        # down leaves every later module pointing at an address that no longer answers --
        # which reads as a preemption or fleet test failing for reasons of its own. Restoring
        # the address the module found is what keeps the failure local to this file.
        ray.shutdown()
        cluster.shutdown()
        if prior_address is not None:
            os.environ["RAY_ADDRESS"] = prior_address
        else:
            os.environ.pop("RAY_ADDRESS", None)

    if ray.cluster_resources().get("CPU", 0) < 4 or _schedulable_nodes() < 2:
        _restore()
        pytest.skip(
            "no two-node cluster to spread a fleet across: the session has fewer than two "
            "CPU-bearing nodes and ray.cluster_utils.Cluster could not simulate one here"
        )
    try:
        yield cluster
    finally:
        _restore()


def _norm(t: pa.Table) -> set:
    return {
        tuple(round(v, 6) if isinstance(v, float) else v for v in r.values()) for r in t.to_pylist()
    }


def test_fleet_spreads_across_nodes(_two_node_cluster):
    """A 4-worker fleet occupies every node it can (SPREAD), so the shuffle isn't pinned.

    The bar is `min(4, nodes)` rather than a literal 2, because the fixture no longer
    guarantees exactly two: on a real cluster this fleet lands on four distinct machines, and
    an `== 2` there fails a *better* placement than the one it was written to demand.
    """
    from batcher.dist.executors.ray_runtime import _ensure_ray, release_placement
    from batcher.dist.flight_worker import spawn_flight_workers

    # The bootstrap every distributed terminal op runs first. It is what decides whether the
    # engine ships itself to the workers, and this test reaches `spawn_flight_workers`
    # directly -- so without it, a fleet on a real cluster dies in `__init__` with
    # `No module named 'batcher'`, where the same fleet raised through `collect(...)` works.
    _ensure_ray(4)
    actors, pg = spawn_flight_workers(4, 4, "")
    try:
        node_ids = ray.get([a.node_id.remote() for a in actors])
        nodes = sum(
            1 for n in ray.nodes() if n.get("Alive") and n.get("Resources", {}).get("CPU", 0) >= 1
        )
        assert len(set(node_ids)) == min(len(actors), nodes), (
            f"fleet did not spread over {nodes} node(s): {node_ids}"
        )
    finally:
        for a in actors:
            ray.kill(a)
        release_placement(pg)


def test_flight_aggregate_correct_on_two_nodes(_two_node_cluster):
    """The placement-group-scheduled Flight aggregate equals single-node when the
    fleet is genuinely spread across two machines."""
    rng = np.random.default_rng(31)
    n = 120_000
    t = pa.table(
        {"k": rng.integers(0, 40, n).astype("int64"), "v": rng.integers(0, 100, n).astype("int64")}
    )

    def q(ds):
        return ds.group_by("k").agg(s=col("v").sum(), n=count())

    single = q(bt.from_arrow(t)).collect()
    flight = q(bt.from_arrow(t)).collect(distributed=True, num_workers=4, transport="flight")
    assert _norm(single) == _norm(flight)
