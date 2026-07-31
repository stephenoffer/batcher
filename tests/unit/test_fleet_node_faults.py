"""The drain list has to include nodes whose GPUs are all fine.

Every entry the fleet health probe had before this was a *device* condition: a quarantined
GPU, a degraded link, a pending reset. That misses the way a node most commonly goes bad on a
GPU cluster, which is not the GPU. The kernel OOM-kills the worker — no exception, no
traceback, the actor is simply gone. Or the spill filesystem is remounted read-only and every
stateful operator placed there fails on its first write.

In both cases every device on the node reads perfectly healthy, the scheduler keeps seeing a
free slot, and the retries walk the whole queue onto it. The node has to appear in the list an
operator drains, and it has to appear for a reason that names the actual cause.
"""

from __future__ import annotations

import pytest

from batcher.dist.executors.ray_runtime.hardware_probe import unhealthy_nodes

pytestmark = pytest.mark.unit


def _clean(node_id: str = "node-a", **overrides) -> dict:
    record = {
        "node_id": node_id,
        "quarantined": [],
        "degraded": [],
        "reset_pending": [],
        "degraded_links": [],
        "nvlink": {},
        "fabric_errors": {},
        "node_faults": {},
        "node_fault_severity": "none",
        "scratch": "ok",
    }
    record.update(overrides)
    return record


def test_a_node_whose_kernel_killed_a_worker_is_drained():
    record = _clean(node_faults={"host_oom": 3}, node_fault_severity="fatal")
    assert unhealthy_nodes((record,)) == (record,)


def test_a_node_that_cannot_write_where_it_spills_is_drained():
    record = _clean(scratch="failed")
    assert unhealthy_nodes((record,)) == (record,)


def test_a_healthy_node_is_not_drained():
    assert unhealthy_nodes((_clean(),)) == ()


def test_a_note_level_kernel_fault_is_not_a_drain_reason():
    # One corrected PCIe error is the link's error correction doing its job. Draining on it
    # would take a working node out for behaving exactly as designed.
    record = _clean(node_faults={"pcie_corrected": 12}, node_fault_severity="note")
    assert unhealthy_nodes((record,)) == ()


def test_low_scratch_space_warns_without_draining():
    assert unhealthy_nodes((_clean(scratch="warn"),)) == ()


def test_a_node_that_could_not_be_checked_is_not_drained():
    # The common case inside a container with no host kernel log. Silence is not evidence,
    # and draining on it would empty a fleet the day a base image changed.
    record = _clean(node_fault_severity="none", scratch="unknown", kernel_log_readable=False)
    assert unhealthy_nodes((record,)) == ()


def test_the_device_reasons_still_drain_a_node():
    # The pre-existing behavior has to survive the addition, not be replaced by it.
    assert unhealthy_nodes((_clean(quarantined=["GPU-abc"]),))
    assert unhealthy_nodes((_clean(degraded_links=["0000:0c:00.0"]),))
