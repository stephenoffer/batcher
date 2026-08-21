"""What a driver learns from its workers, and what it must not learn from itself.

Two learned quantities are in *machine units* and are fitted on the driver: the cost
coefficients (`kyber.calibration`) and the per-family CPU utilization that sizes a distributed
task's `num_cpus` (`kyber.cpu_shares`). Both read `MetadataHub.op_stats_by_kind`.

That view used to be filtered to rows this process measured. The rows arrive correctly
attributed — a worker stamps its own hardware fingerprint before shipping its metrics, and the
driver's transcription preserves it — so on any cluster whose driver is a different machine
class from its workers, every worker measurement was dropped at the reader. No error, no
warning: the coefficients simply stayed at their shipped defaults and every operator family
silently kept its static CPU prior, on precisely the deployment the CPU-share loop exists for.

The view is now bucketed per machine class instead of filtered to one, and the caller names the
class it is planning *for*. These tests pin both halves: a worker's rows are readable under the
worker's key, and they are still not readable under anybody else's.
"""

from __future__ import annotations

import pytest

from batcher._internal.hardware import fingerprint
from batcher.config import active_config
from batcher.kyber.calibration import calibrate
from batcher.kyber.cpu_shares import load_cpu_utilization
from batcher.metadata import MetadataHub
from batcher.metadata.backends.in_process import InProcessBackend
from batcher.plan.feedback import OperatorFeedback, OpId

pytestmark = pytest.mark.unit

# Two machine classes that are definitionally not this process's.
_WORKER = "ffffffffffff"
_OTHER = "aaaaaaaaaaaa"


def _hub(machine: str | None, *, kinds=("aggregate", "filter"), count=64) -> MetadataHub:
    """A hub holding `count` operators per kind, all measured on machine class `machine`."""
    hub = MetadataHub(InProcessBackend())
    op = 0
    for kind in kinds:
        for _ in range(count):
            op += 1
            extra = {"hw_fingerprint": machine} if machine else {}
            hub.record(
                OperatorFeedback(
                    op_id=OpId(op),
                    kind=kind,
                    n_actual=1_000_000,
                    t_op_ms=900.0 if kind == "aggregate" else 9.0,
                    m_peak_bytes=1 << 26,
                    selectivity=1.0,
                    batch_size=16_384,
                    n_input=1_000_000,
                    signature=f"sig-{kind}",
                    n_estimated=1e6,
                    cpu_utilization=0.95,
                    threads=8,
                    **extra,
                )
            )
    return hub


def test_a_workers_rows_are_invisible_under_the_drivers_key():
    """The scoping property that must survive: unlike machines do not blend."""
    hub = _hub(_WORKER)
    assert hub.op_stats_by_kind() == {}
    assert hub.op_stats_by_kind(_OTHER) == {}


def test_a_workers_rows_are_visible_under_the_workers_key():
    """The defect, stated as the case it produces. This is what used to be unreachable."""
    hub = _hub(_WORKER)
    by_kind = hub.op_stats_by_kind(_WORKER)
    assert set(by_kind) == {"aggregate", "filter"}
    assert len(by_kind["aggregate"]) == 64


def test_machine_classes_do_not_leak_into_each_other():
    """Two classes writing one store stay separate, which is the whole point of the key."""
    hub = MetadataHub(InProcessBackend())
    for i, machine in enumerate((_WORKER, _OTHER)):
        hub.record(
            OperatorFeedback(
                op_id=OpId(i),
                kind="scan",
                n_actual=10,
                t_op_ms=1.0,
                m_peak_bytes=0,
                selectivity=1.0,
                batch_size=1,
                hw_fingerprint=machine,
            )
        )
    assert len(hub.op_stats_by_kind(_WORKER)["scan"]) == 1
    assert len(hub.op_stats_by_kind(_OTHER)["scan"]) == 1


def test_a_row_with_no_fingerprint_belongs_to_no_machine():
    """An unattributed row is not evidence about any machine, including this one.

    Rows written before the field existed carry no class. Adopting them into whichever class
    asked first would reinstate exactly the blend the scoping removes.
    """
    hub = MetadataHub(InProcessBackend())
    hub.record(
        OperatorFeedback(
            op_id=OpId(1),
            kind="scan",
            n_actual=10,
            t_op_ms=1.0,
            m_peak_bytes=0,
            selectivity=1.0,
            batch_size=1,
            hw_fingerprint="",
        )
    )
    assert hub.op_stats_by_kind() == {}
    assert hub.op_stats_by_kind("") == {}
    assert hub.op_stats_by_kind(_WORKER) == {}


def test_calibration_fits_from_the_target_machines_rows():
    """The coefficients move when read for the workers, and not when read for the driver."""
    cfg = active_config()
    hub = _hub(_WORKER)
    defaults = cfg.optimizer.cost_coeffs
    assert calibrate(hub, cfg) == defaults, "the driver measured nothing and must fit nothing"
    assert calibrate(hub, cfg, _WORKER) != defaults, "the workers measured plenty"


def test_calibration_does_not_serve_one_machine_from_anothers_cache():
    """The fit is memoized per hub, so the machine class has to be part of that key.

    A session that plans single-node and distributed in turn asks the same hub for two
    different answers; without the class in the key the second call returns the first's.
    """
    cfg = active_config()
    hub = _hub(_WORKER)
    worker_fit = calibrate(hub, cfg, _WORKER)
    driver_fit = calibrate(hub, cfg)
    assert worker_fit != driver_fit
    assert calibrate(hub, cfg, _WORKER) == worker_fit


def test_cpu_shares_learn_from_the_workers_that_will_run_the_task():
    """The loop whose entire purpose is sizing a distributed task's `num_cpus`."""
    cfg = active_config()
    hub = _hub(_WORKER)
    assert load_cpu_utilization(hub, cfg) == {}
    learned = load_cpu_utilization(hub, cfg, _WORKER)
    assert learned["aggregate"] == pytest.approx(0.95)


def test_a_single_node_run_reads_its_own_measurements_unchanged():
    """The default path, which is what every caller without a cluster profile takes."""
    cfg = active_config()
    hub = _hub(None)  # stamped by `OperatorFeedback`'s own default: this machine
    assert set(hub.op_stats_by_kind()) == {"aggregate", "filter"}
    assert hub.op_stats_by_kind(fingerprint()) == hub.op_stats_by_kind()
    assert load_cpu_utilization(hub, cfg)["aggregate"] == pytest.approx(0.95)
    assert calibrate(hub, cfg) != cfg.optimizer.cost_coeffs


def test_a_mixed_fleet_falls_back_to_the_local_class():
    """`""` means "no single honest answer", and must read as the pre-existing behavior.

    `HardwareProfile.fingerprint` is `""` on a fleet whose workers disagree, following the same
    rule `accelerator_type` does. The consumers pass `None` in that case, so they degrade to
    what they did before this existed rather than picking one worker's class arbitrarily.
    """
    hub = _hub(None)
    assert hub.op_stats_by_kind("") == hub.op_stats_by_kind()


def test_the_planner_contract_carries_the_class_it_plans_for():
    """`HardwareProfile` is where the class travels from the probe to the consumers."""
    from batcher.plan.resource import HardwareProfile

    assert HardwareProfile.local().fingerprint == fingerprint()
    # A cluster whose workers could not be probed reports no opinion, not the driver's.
    assert (
        HardwareProfile.for_cluster(cpu_cores=8, memory_bytes=1, worker_count=4).fingerprint == ""
    )
    assert (
        HardwareProfile.for_cluster(
            cpu_cores=8, memory_bytes=1, worker_count=4, fingerprint=_WORKER
        ).fingerprint
        == _WORKER
    )


def test_op_stats_by_kind_reads_the_class_the_scope_is_planning_for() -> None:
    """The default resolves through `planning_for`, not straight to this process.

    A distributed run wraps its whole plan-admit-execute span in `planning_for(workers)` so a
    read and a write cannot key the same learned quantity differently. Two consumers of this
    view pass no class at all — `carbonite.memory.learned`, which fits the memory model behind
    admission and the per-task grant, and `dist.adaptive_sizing`, whose subject is how to shape
    a task on a worker — so a default that ignored the scope handed both the *driver's* rows.
    On the ordinary Ray shape (a small head node, large workers) that is an empty view, and
    both silently kept their cold-start defaults on the deployment they exist for.
    """
    from batcher.metadata.hardware_scope import planning_for

    hub = MetadataHub(InProcessBackend())
    hub.record(
        OperatorFeedback(
            op_id=OpId(1),
            kind="aggregate",
            n_actual=10,
            t_op_ms=1.0,
            m_peak_bytes=1024,
            selectivity=1.0,
            batch_size=16_384,
            hw_fingerprint="worker-class",
        )
    )
    assert hub.op_stats_by_kind() == {}, "the driver measured nothing, and says so"
    with planning_for("worker-class"):
        assert list(hub.op_stats_by_kind()) == ["aggregate"]
    # An explicit class still wins over the scope, so every existing caller is unaffected.
    with planning_for("worker-class"):
        assert hub.op_stats_by_kind("some-other-class") == {}
