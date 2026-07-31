"""The gang a GPU collective reserves is the one `plan_collective` laid out, not W identical
bundles.

`plan_collective` had no caller: it knew to keep a collective inside one coherent domain, to
fill the largest domain first when it cannot, and to skip a node a residency rule or a power
budget excluded — and the scheduler reserved uniform bundles and discovered none of it. These
pin the wiring, and every refusal that keeps it from reshaping a gang it does not understand.
"""

from __future__ import annotations

import pytest

from batcher.dist.executors.ray_runtime.fabric import CollectivePlacement
from batcher.dist.executors.ray_runtime.scheduling import _collective_bundles
from batcher.plan.resource import SchedulingEnvelope

pytestmark = pytest.mark.unit


def _env(**kwargs) -> SchedulingEnvelope:
    fields = {"n_tasks": 4, "num_cpus": 2.0, "num_gpus": 1.0, "gpu_collective": True}
    fields.update(kwargs)
    return SchedulingEnvelope(**fields)


def _placed(monkeypatch, placement: CollectivePlacement) -> None:
    import batcher.dist.executors.ray_runtime.fabric as fabric

    monkeypatch.setattr(fabric, "plan_collective", lambda *a, **k: placement, raising=True)


def test_a_collective_reserves_the_planned_layout(monkeypatch) -> None:
    _placed(
        monkeypatch,
        CollectivePlacement(
            world_size=4, bundles=({"GPU": 4.0, "CPU": 8.0},), strategy="STRICT_PACK"
        ),
    )
    assert _collective_bundles(4, _env(), {}) == [{"GPU": 4.0, "CPU": 8.0}]


def test_a_split_layout_survives_as_several_bundles(monkeypatch) -> None:
    _placed(
        monkeypatch,
        CollectivePlacement(
            world_size=4,
            bundles=({"GPU": 2.0, "CPU": 4.0}, {"GPU": 2.0, "CPU": 4.0}),
            strategy="PACK",
        ),
    )
    assert len(_collective_bundles(4, _env(), {})) == 2


def test_a_short_plan_falls_back_rather_than_reserving_a_partial_gang(monkeypatch) -> None:
    """A partial gang reserves successfully, then hangs on a world size it never receives."""
    _placed(
        monkeypatch,
        CollectivePlacement(world_size=4, bundles=({"GPU": 2.0, "CPU": 4.0},), strategy="PACK"),
    )
    assert _collective_bundles(4, _env(), {}) is None


def test_an_unreadable_topology_keeps_the_uniform_bundles(monkeypatch) -> None:
    _placed(monkeypatch, CollectivePlacement(world_size=4))
    assert _collective_bundles(4, _env(), {}) is None


def test_a_stage_that_is_not_a_collective_is_untouched() -> None:
    assert _collective_bundles(4, _env(gpu_collective=False), {}) is None


def test_a_stage_with_several_devices_per_worker_keeps_its_own_shape(monkeypatch) -> None:
    """The plan's bundles carry devices, not worker slots; reshaping them changes the width."""
    _placed(
        monkeypatch,
        CollectivePlacement(
            world_size=4, bundles=({"GPU": 4.0, "CPU": 8.0},), strategy="STRICT_PACK"
        ),
    )
    assert _collective_bundles(4, _env(num_gpus=2.0), {}) is None


def test_a_single_worker_needs_no_gang() -> None:
    assert _collective_bundles(1, _env(), {}) is None
    assert _collective_bundles(4, None, {}) is None


def test_the_node_class_restriction_is_merged_into_every_bundle(monkeypatch) -> None:
    """A CPU-only restriction outside the bundle reserves nothing, so it has to be inside it."""
    _placed(
        monkeypatch,
        CollectivePlacement(
            world_size=4,
            bundles=({"GPU": 2.0, "CPU": 4.0}, {"GPU": 2.0, "CPU": 4.0}),
            strategy="PACK",
        ),
    )
    bundles = _collective_bundles(4, _env(), {"batcher_gpu_node": 0.001})
    assert all(b["batcher_gpu_node"] == 0.001 for b in bundles)


def test_memory_scales_with_the_devices_in_a_bundle(monkeypatch) -> None:
    _placed(
        monkeypatch,
        CollectivePlacement(
            world_size=4, bundles=({"GPU": 4.0, "CPU": 8.0},), strategy="STRICT_PACK"
        ),
    )
    bundles = _collective_bundles(4, _env(memory_bytes=1_000_000), {})
    assert bundles[0]["memory"] == 4_000_000


def test_a_planning_failure_never_fails_the_placement(monkeypatch) -> None:
    import batcher.dist.executors.ray_runtime.fabric as fabric

    def boom(*_a, **_k):
        raise RuntimeError("topology read failed")

    monkeypatch.setattr(fabric, "plan_collective", boom, raising=True)
    assert _collective_bundles(4, _env(), {}) is None
