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
    import os

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
    try:
        yield cluster
    finally:
        # Tear the private cluster down *and leave the process able to find the shared one
        # again*. `ray.init` records its address in the environment, so a module that
        # replaces the session's cluster with a 2-node, 4-CPU one and then only shuts it
        # down leaves every later module pointing at an address that no longer answers —
        # which reads as a preemption or fleet test failing for reasons of its own. Restoring
        # the address the module found is what keeps the failure local to this file.
        ray.shutdown()
        cluster.shutdown()
        if prior_address is not None:
            os.environ["RAY_ADDRESS"] = prior_address
        else:
            os.environ.pop("RAY_ADDRESS", None)


def _norm(t: pa.Table) -> set:
    return {
        tuple(round(v, 6) if isinstance(v, float) else v for v in r.values()) for r in t.to_pylist()
    }


def test_fleet_spreads_across_nodes(_two_node_cluster):
    """A 4-worker fleet on a 2-node cluster lands on both nodes (SPREAD), so the
    shuffle isn't pinned to one machine."""
    from batcher.dist.executors.ray_runtime import release_placement
    from batcher.dist.flight_worker import spawn_flight_workers

    actors, pg = spawn_flight_workers(4, 4, "")
    try:
        node_ids = ray.get([a.node_id.remote() for a in actors])
        assert len(set(node_ids)) == 2, f"fleet did not spread: {node_ids}"
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
