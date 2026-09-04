"""The GPU fan-out must be tiled by devices that work, not devices that exist.

Carbonite's `gpu_envelope` clamps a GPU grant three ways: to the devices that exist, to the
devices the power budget can run, and to the devices that pass the health verdicts. The
distributed executor then *replaces* that `n_tasks` with `_accelerator_fill_workers`, which
recomputes the fan-out per node so it can also size the per-worker core grant — and it used to
count **nameplate** devices, silently discarding two of those three ceilings.

That is not a throughput point. A device rarely fails by disappearing: it stays present
reporting uncorrectable ECC errors, or the driver clamps it to a fraction of its clock, and a
fan-out sized by the raw count keeps placing actors on it. The health machinery
(`accelerator.health`, `schedulable_device_count`, `unhealthy_gpus_by_node`) existed and was
being computed and thrown away on the one path that decides where GPU work lands.

These run against an injected node census; nothing here needs a cluster or a device.
"""

from __future__ import annotations

import dataclasses

import pytest

from batcher.config import Config, config_context

pytestmark = pytest.mark.unit


def _node(cpus: float, gpus: float, *, healthy: float | None = None, model: str | None = None):
    """One `node_classes()` row as the fan-out reads it."""
    row = {"cpus": cpus, "gpus": gpus, "accelerator_type": model}
    if healthy is not None:
        row["healthy_gpus"] = healthy
    return row


def _fill(monkeypatch, classes, num_gpus: float):
    import batcher.dist.executors.ray_runtime.scaling as scaling
    from batcher.dist import executor

    monkeypatch.setattr(scaling, "node_classes", lambda: classes)
    return executor._accelerator_fill_workers(num_gpus)


# --- the projection carries what the fan-out needs -------------------------------------------


def test_the_census_projects_device_health_through_to_placement():
    """`_class_entry` is what every placement consumer reads. The census measured device
    health all along and this projection dropped it, which is why the fan-out could not see
    it."""
    from batcher.dist.executors.ray_runtime.fabric.census import build_census
    from batcher.dist.executors.ray_runtime.scaling import _class_entry

    nodes = [{"Alive": True, "NodeID": "a", "Resources": {"CPU": 8.0, "GPU": 4.0}, "Labels": {}}]
    (klass,) = build_census(nodes, None, {"a": 3}, shape_zone=lambda _l: "")
    entry = _class_entry(klass)
    assert entry["gpus"] == 4.0, "the nameplate count stays available"
    assert entry["healthy_gpus"] == 1.0, "three of the four are quarantined"


def test_an_unprobed_fleet_reports_every_device_healthy():
    """An absent probe is not evidence that a fleet is unhealthy. Reading it as one would take
    a cluster offline the day telemetry stopped being installed."""
    from batcher.dist.executors.ray_runtime.fabric.census import build_census
    from batcher.dist.executors.ray_runtime.scaling import _class_entry

    nodes = [{"Alive": True, "NodeID": "a", "Resources": {"CPU": 8.0, "GPU": 4.0}, "Labels": {}}]
    (klass,) = build_census(nodes, None, {}, shape_zone=lambda _l: "")
    assert _class_entry(klass)["healthy_gpus"] == 4.0


# --- the fan-out uses it ---------------------------------------------------------------------


def test_quarantined_devices_do_not_get_workers(monkeypatch):
    """The defect, stated directly: two nodes of four devices with three quarantined between
    them must host five workers, not eight."""
    classes = [_node(32.0, 4.0, healthy=3.0), _node(32.0, 4.0, healthy=2.0)]
    workers, num_cpus = _fill(monkeypatch, classes, 1.0)
    assert workers == 5
    assert num_cpus >= 1.0


def test_a_healthy_fleet_is_sized_exactly_as_before(monkeypatch):
    """The control. Without it the change is satisfied by any fan-out that is merely smaller,
    including one that broke the device tiling outright."""
    classes = [_node(32.0, 4.0, healthy=4.0), _node(32.0, 4.0, healthy=4.0)]
    assert _fill(monkeypatch, classes, 1.0) == (8, 8.0)
    # And a fractional request still packs several workers per device.
    assert _fill(monkeypatch, classes, 0.5)[0] == 16


def test_a_projection_without_health_keeps_the_nameplate_count(monkeypatch):
    """A census that predates the field must size the fleet the way it always did, not as
    though every device were unhealthy."""
    classes = [_node(32.0, 4.0), _node(32.0, 4.0)]
    assert _fill(monkeypatch, classes, 1.0) == (8, 8.0)


def test_a_node_whose_devices_are_all_quarantined_hosts_nothing(monkeypatch):
    """It cannot run this stage, so it must not contribute workers — nor drag the per-worker
    core grant down by being counted as a host."""
    classes = [_node(8.0, 4.0, healthy=0.0), _node(64.0, 4.0, healthy=4.0)]
    workers, num_cpus = _fill(monkeypatch, classes, 1.0)
    assert workers == 4
    assert num_cpus == 16.0, "the grant comes from the surviving host, not the dead one"


def test_a_fleet_with_no_healthy_device_declines_the_device_fill(monkeypatch):
    """`None` hands the caller back to its existing sizing rather than inventing a fan-out."""
    classes = [_node(32.0, 4.0, healthy=0.0)]
    assert _fill(monkeypatch, classes, 1.0) is None


# --- the power ceiling is reapplied, through Carbonite's own helper --------------------------


def test_the_power_budget_is_a_no_op_when_none_is_configured(monkeypatch):
    """Off by default: an operator who has not said what the rack can draw gets inventory."""
    classes = [_node(64.0, 8.0, healthy=8.0, model="NVIDIA_A100")]
    assert _fill(monkeypatch, classes, 1.0)[0] == 8


def test_the_power_budget_clamps_the_device_fan_out(monkeypatch):
    """A rack whose busway cannot power every slot does not trip a breaker -- it clamps every
    device in the zone, which reads as the whole rack getting slower for no visible reason.
    Carbonite refuses that fan-out for a GPU envelope; the device fill has to refuse it too,
    or the clamp is computed and then overwritten."""
    from batcher.carbonite.accel.power import devices_within_budget

    cfg = Config()
    budgeted = cfg.replace(
        accelerator=dataclasses.replace(
            cfg.accelerator,
            energy=dataclasses.replace(cfg.accelerator.energy, power_budget_watts=1200.0),
        )
    )
    classes = [_node(64.0, 8.0, healthy=8.0, model="NVIDIA_A100")]
    with config_context(budgeted):
        allowed = devices_within_budget("NVIDIA_A100", 8)
        # The control: this test means nothing unless the budget actually binds here.
        assert allowed < 8, "pick a budget that clamps, or this asserts nothing"
        assert _fill(monkeypatch, classes, 1.0)[0] == allowed


def test_a_mixed_model_fleet_is_not_priced_by_whichever_model_came_first(monkeypatch):
    """Two device models have two draw figures. Pricing the fleet by the first listed would
    under-count a fleet of small parts and over-count one of large parts, and the second is the
    direction that clamps a rack -- so the budget declines instead of guessing."""
    cfg = Config()
    budgeted = cfg.replace(
        accelerator=dataclasses.replace(
            cfg.accelerator,
            energy=dataclasses.replace(cfg.accelerator.energy, power_budget_watts=1200.0),
        )
    )
    classes = [
        _node(64.0, 8.0, healthy=8.0, model="NVIDIA_A100"),
        _node(64.0, 8.0, healthy=8.0, model="NVIDIA_TESLA_T4"),
    ]
    with config_context(budgeted):
        assert _fill(monkeypatch, classes, 1.0)[0] == 16, "inventory, not a guessed budget"
