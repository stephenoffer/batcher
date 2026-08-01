"""Admitting GPU work on a cluster: what the gate asks for, and what "full" means.

Both defects here are the same mistake seen twice — a check that was written against a fact
about the *worker*, and outlived it:

* the admission gate required a free CPU because a GPU shard task once took Ray's default
  `num_cpus=1`. It no longer does, and asking for a resource the task does not request can only
  refuse work that would have run — on precisely the cluster (a shuffle fleet holding every
  core) that the zero-CPU request was introduced to rescue;
* the health check called a device full on its *total* residency, so a worker that had grown
  its own memory pool quarantined the device it was computing on.

Neither needs a GPU to reproduce, and neither could be reproduced on one machine running one
process, which is why both survived.
"""

from __future__ import annotations

import sys
import types

import pytest

from batcher.carbonite.accel.health import HealthThresholds, assess_device
from batcher.dist.gpu import dispatch
from batcher.dist.gpu.shards import is_memory_failure

pytestmark = pytest.mark.unit

GIB = 1 << 30


@pytest.fixture
def connected(monkeypatch):
    """A `ray` that reports itself connected, so the admission loop is actually entered.

    Without it every assertion in this file passes for the wrong reason: `await_gpu_admission`
    returns True immediately on an unconnected process, so a test that asserts True is true by
    construction and one that asserts False can never pass. That is the failure mode
    `lint-tests` exists for, and it is worth saying out loud because the gate under test is
    itself a "return True when you cannot tell" gate — the shape most likely to be tested
    vacuously.
    """
    fake = types.ModuleType("ray")
    fake.is_initialized = lambda: True
    fake.available_resources = dict
    monkeypatch.setitem(sys.modules, "ray", fake)
    return fake


def _telemetry(**kwargs):
    from batcher._internal.hardware.nvml import DeviceTelemetry

    base = {"index": 0, "memory_total_bytes": 80 * GIB, "memory_used_bytes": 0}
    return DeviceTelemetry(**{**base, **kwargs})


def test_a_device_free_of_cores_still_admits_a_gpu_stage(monkeypatch, connected):
    """The regression: a shuffle fleet holding every CPU must not refuse a GPU task that asks
    for none. Left in, the gate turned a fixed deadlock into a full-budget stall followed by a
    silent fall back to the CPU engine, with every device idle."""
    monkeypatch.setattr(dispatch, "_free_resources", lambda: {"GPU": 4.0, "CPU": 0.0})
    assert dispatch.await_gpu_admission() is True


def test_a_fleet_with_no_free_device_is_refused(monkeypatch, connected):
    """The check the gate does exist for still fires."""
    monkeypatch.setattr(dispatch, "_free_resources", lambda: {"GPU": 0.0, "CPU": 64.0})
    from batcher.config import option_context

    with option_context("distributed.gpu_admission_wait_s", 0.01):
        assert dispatch.await_gpu_admission() is False


def test_a_packed_share_is_admitted_by_a_partly_busy_device(monkeypatch, connected):
    """A fan-out asking for a quarter of a device is admitted by one three-quarters busy."""
    monkeypatch.setattr(dispatch, "_free_resources", lambda: {"GPU": 0.25, "CPU": 0.0})
    assert dispatch.await_gpu_admission(0.25) is True


def test_an_unreadable_cluster_admits_rather_than_refuses(monkeypatch, connected):
    """Refusing on an unreadable cluster would disable the GPU backend on every deployment
    whose resource view differs from the one this was written against."""
    monkeypatch.setattr(dispatch, "_free_resources", lambda: None)
    assert dispatch.await_gpu_admission() is True


def test_a_device_this_process_filled_itself_is_not_quarantined():
    """A worker that grew its own pool into the memory it was entitled to must not conclude the
    device is somebody else's. This is what made a busy fleet report itself sick."""
    reading = _telemetry(memory_used_bytes=76 * GIB)  # 95% resident, all of it ours
    verdict = assess_device(reading, HealthThresholds(), own_bytes=76 * GIB)
    assert verdict.state == "healthy"
    assert "memory_full" not in verdict.reasons


def test_a_device_a_neighbour_filled_is_still_quarantined():
    """The condition the threshold exists for is unchanged: someone else's 96% is a device this
    stage must not be admitted onto."""
    reading = _telemetry(memory_used_bytes=77 * GIB)
    verdict = assess_device(reading, HealthThresholds(), own_bytes=0)
    assert "memory_full" in verdict.reasons
    assert verdict.state == "quarantine"


def test_a_shared_device_is_judged_on_the_neighbours_share_alone():
    """Half ours and half theirs is not a full device, however the total reads."""
    reading = _telemetry(memory_used_bytes=78 * GIB)
    verdict = assess_device(reading, HealthThresholds(), own_bytes=40 * GIB)
    assert "memory_full" not in verdict.reasons


def test_an_unattributed_process_keeps_the_old_conservative_reading():
    """`own_bytes=0` is what a caller that cannot attribute passes, and it must behave exactly
    as this did before — the whole residency counted as external."""
    reading = _telemetry(memory_used_bytes=79 * GIB)
    assert "memory_full" in assess_device(reading, HealthThresholds()).reasons


@pytest.mark.parametrize(
    "message",
    [
        "RayTaskError(RMMError): Maximum pool size exceeded",
        "std::bad_alloc: out_of_memory",
        "CUDA error: out of memory",
        "cudaErrorMemoryAllocation",
        "Failed to allocate 4194304 bytes on device",
    ],
)
def test_every_way_a_device_reports_exhaustion_takes_the_subdivision_ladder(message):
    """A bounded RMM pool refuses with wording none of the original markers matched, and Ray
    strips the exception type that would otherwise have carried it — so the one overflow a
    configured pool produces *by design* was the one the ladder did not recognize. It read as a
    deterministic error and the shard went straight to the CPU engine."""
    assert is_memory_failure(RuntimeError(message))


def test_a_deterministic_error_is_still_not_retried_in_pieces():
    """Subdividing a deterministic failure pays N times to reach the same conclusion."""
    assert not is_memory_failure(KeyError("no column named 'total'"))


def test_an_undividable_shard_keeps_the_overflow_that_caused_it():
    """The caller falls back to the host on this error and reports it as the reason; a bare
    `MemoryError` erased the device-side account of why a GPU query ran on the CPU."""
    from batcher.dist.gpu.shards import run_subdivided

    cause = MemoryError("std::bad_alloc: out_of_memory [bt-device-peak 100/50]")
    with pytest.raises(MemoryError) as caught:
        run_subdivided(
            {"splits": ["only-one"]},
            lambda piece: None,
            parts=2,
            rounds=2,
            cause=cause,
        )
    assert caught.value.__cause__ is cause
