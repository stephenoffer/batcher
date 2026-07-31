"""A stage that holds a device must be tiled by devices, not by cores.

`_cluster_fill_workers` sizes the fan-out at one worker per core-slice of a node, and its
reasoning is right for a relational query: more workers than nodes cannot add CPU parallelism
because cores are the limit. For a stage that holds an accelerator the limit is *devices*, and
the core-shaped fill caps the fan-out at the node count however many devices a node has. On the
ordinary four-devices-per-node shape that stranded three quarters of the fleet -- measured on a
real 4-node, 16-GPU cluster, an inference stage was given 4 workers and ran on 2 devices.

These tests pin the device-shaped tiling and, just as importantly, that it declines to act on
anything that is not an accelerator stage, so the core-shaped fill still decides every
relational query exactly as it did.
"""

from __future__ import annotations

import pytest

from batcher.dist.executor import _accelerator_fill_workers

pytestmark = pytest.mark.unit


def _classes(*nodes: tuple[float, float]) -> list[dict]:
    """`node_classes()`-shaped entries from `(cpus, gpus)` pairs."""
    return [
        {"cpus": c, "gpus": g, "memory": 1 << 37, "accelerators": 0.0, "accelerator_type": "T4"}
        for c, g in nodes
    ]


@pytest.fixture
def fleet(monkeypatch):
    """Install a fake cluster shape behind `node_classes`."""

    def install(*nodes: tuple[float, float]) -> None:
        from batcher.dist.executors.ray_runtime import scaling

        monkeypatch.setattr(scaling, "node_classes", lambda: _classes(*nodes))

    return install


def test_this_repos_own_fleet_is_tiled_by_its_devices(fleet) -> None:
    """Four nodes of 48 cores and 4 T4s: 16 workers of one device and 12 cores."""
    fleet((48.0, 4.0), (48.0, 4.0), (48.0, 4.0), (48.0, 4.0))
    assert _accelerator_fill_workers(1.0) == (16, 12.0)


def test_a_whole_device_per_worker_uses_every_device(fleet) -> None:
    fleet((64.0, 8.0), (64.0, 8.0))
    workers, cores = _accelerator_fill_workers(1.0)
    assert workers == 16, "16 devices, 16 workers"
    assert cores == 8.0, "each node's cores split among the workers it hosts"


def test_a_multi_device_worker_gets_fewer_workers(fleet) -> None:
    """A stage wanting two devices each halves the worker count and doubles the cores."""
    fleet((48.0, 4.0), (48.0, 4.0))
    assert _accelerator_fill_workers(2.0) == (4, 24.0)


def test_a_fractional_device_grant_packs_several_workers_per_device(fleet) -> None:
    fleet((48.0, 4.0), (48.0, 4.0))
    workers, cores = _accelerator_fill_workers(0.5)
    assert workers == 16, "eight half-device workers per node"
    assert cores == 6.0


def test_a_non_accelerator_stage_declines(fleet) -> None:
    """The core-shaped fill must keep deciding every relational query."""
    fleet((48.0, 4.0), (48.0, 4.0))
    assert _accelerator_fill_workers(0.0) is None
    assert _accelerator_fill_workers(-1.0) is None


def test_a_fleet_with_no_devices_declines(fleet) -> None:
    fleet((48.0, 0.0), (48.0, 0.0))
    assert _accelerator_fill_workers(1.0) is None


def test_cpu_only_nodes_host_nothing_but_do_not_shrink_the_grant(fleet) -> None:
    """The mixed shape this cluster actually is: CPU nodes beside accelerator nodes.

    A CPU-only node cannot run a stage that holds a device, so it contributes no workers --
    and, critically, its cores must not drag the per-worker grant around either.
    """
    fleet((96.0, 0.0), (48.0, 4.0), (48.0, 4.0))
    assert _accelerator_fill_workers(1.0) == (8, 12.0)


def test_a_device_denser_node_hosts_more_workers(fleet) -> None:
    fleet((48.0, 4.0), (96.0, 8.0))
    workers, cores = _accelerator_fill_workers(1.0)
    assert workers == 12, "4 + 8 devices"
    assert cores == 12.0, "both nodes give 12 cores per device"


def test_the_grant_fits_the_tightest_accelerator_node(fleet) -> None:
    """A worker must be placeable on every node that hosts one, so the grant takes the min."""
    fleet((16.0, 4.0), (96.0, 4.0))
    workers, cores = _accelerator_fill_workers(1.0)
    assert workers == 8
    assert cores == 4.0, "16 cores over 4 devices is the binding node"


def test_a_single_worker_answer_declines(fleet) -> None:
    """One worker is what the caller would have done anyway; leave it to the existing path."""
    fleet((48.0, 1.0))
    assert _accelerator_fill_workers(1.0) is None


def test_a_grant_larger_than_any_node_declines(fleet) -> None:
    """Eight devices per worker on four-device nodes: no node can host one."""
    fleet((48.0, 4.0), (48.0, 4.0))
    assert _accelerator_fill_workers(8.0) is None


def test_the_grant_is_never_a_fraction_of_a_core(fleet) -> None:
    """A device-dense, core-poor node must still ask for a whole core."""
    fleet((2.0, 8.0), (2.0, 8.0))
    workers, cores = _accelerator_fill_workers(1.0)
    assert workers == 16
    assert cores == 1.0


def test_an_unreadable_topology_declines(monkeypatch) -> None:
    """Never let a probe failure change the schedule."""
    from batcher.dist.executors.ray_runtime import scaling

    def boom():
        raise RuntimeError("ray is down")

    monkeypatch.setattr(scaling, "node_classes", boom)
    assert _accelerator_fill_workers(1.0) is None
