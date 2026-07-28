"""Learned state is scoped to the machine that measured it — and only where it should be.

The failure this prevents is silent and has no symptom at the point of failure. Two unlike
machines sharing one metadata store fit a single model from both their measurements; the model
is wrong for each, no error is raised, and the only visible effect is that plans are worse than
they were. It shows up wherever a store is shared: a heterogeneous Ray cluster, an autoscaling
group spanning instance generations, a laptop and CI on one checkout.

The fix is a hardware fingerprint in the key, and the risk in applying it is over-application:
scoping a *data* statistic fragments the statistics that take longest to collect and makes a
well-calibrated fleet into N poorly-calibrated ones. So both directions are tested here — what
must be scoped, and what must never be.
"""

from __future__ import annotations

import dataclasses

import pytest

from batcher._internal import hardware
from batcher.metadata.backends import InProcessBackend
from batcher.metadata.hardware_scope import scoped, scoped_key
from batcher.metadata.hub import MetadataHub
from batcher.plan.feedback import OperatorFeedback

pytestmark = pytest.mark.unit


def _feedback(kind: str, *, fingerprint: str, ms: float, op_id: int = 0) -> OperatorFeedback:
    """One operator feedback row attributed to a given machine."""
    return OperatorFeedback(
        op_id=op_id,
        kind=kind,
        n_actual=1_000,
        t_op_ms=ms,
        m_peak_bytes=4096,
        selectivity=1.0,
        batch_size=16_384,
        n_input=1_000,
        signature=f"sig-{kind}",
        n_estimated=1_000.0,
        hw_fingerprint=fingerprint,
    )


def test_scoped_qualifies_a_namespace_with_this_machine():
    name = scoped("kyber.cost")
    assert name.startswith("kyber.cost@")
    assert name.endswith(hardware.fingerprint())
    # A scoped name can never collide with an unscoped one, so old and new entries coexist
    # rather than one silently shadowing the other.
    assert name != "kyber.cost"
    assert scoped_key("model-a").endswith(hardware.fingerprint())


def test_scoping_is_stable_within_a_machine():
    # Instability here would be worse than no scoping: every process would write to a fresh
    # namespace and nothing would ever accumulate enough samples to be used.
    assert scoped("ns") == scoped("ns")
    hardware.reset_hardware_probes()
    assert scoped("ns") == scoped("ns")


def test_cost_feedback_is_filtered_to_this_machine(monkeypatch):
    # `op_stats_by_kind` feeds cost calibration, the memory model, the CPU-share model, and
    # distributed sizing — every one of them fitted in machine units. A row from another
    # machine in this view is a data point about hardware the query will not run on.
    hub = MetadataHub(InProcessBackend())
    here = hardware.fingerprint()
    hub.record(_feedback("hash_join", fingerprint=here, ms=10.0))
    hub.record(_feedback("hash_join", fingerprint="deadbeef0000", ms=1000.0, op_id=1))
    hub.record(_feedback("hash_join", fingerprint="", ms=5000.0, op_id=2))

    rows = hub.op_stats_by_kind().get("hash_join", [])
    assert [r["t_op_ms"] for r in rows] == [10.0], (
        "only this machine's measurements may reach the cost model; a 100x-slower machine's "
        "row would drag the fitted per-row coefficient with it"
    )


def test_cardinality_feedback_is_not_filtered():
    # `op_stats_with_signature` feeds cardinality correction. A filter's selectivity and a
    # join's fan-out are properties of the *data* and are identical on every machine, so
    # scoping this view would discard usable evidence and slow convergence for no benefit.
    hub = MetadataHub(InProcessBackend())
    hub.record(_feedback("filter", fingerprint=hardware.fingerprint(), ms=1.0))
    hub.record(_feedback("filter", fingerprint="deadbeef0000", ms=1.0, op_id=1))

    signed = hub.op_stats_with_signature()
    assert len(signed) == 2, (
        "cardinality evidence must be shared across machines — it describes the data, "
        "not the hardware"
    )


def test_a_row_from_an_unknown_machine_is_not_adopted():
    # Rows written before the fingerprint existed carry `""`. "Measured on an unknown machine"
    # is not evidence about this one, and adopting them would reintroduce exactly the blend
    # the scoping removes — on the very first run after an upgrade.
    hub = MetadataHub(InProcessBackend())
    hub.record(_feedback("sort", fingerprint="", ms=42.0))
    assert hub.op_stats_by_kind().get("sort", []) == []


def test_the_incremental_fold_applies_the_same_filter_as_the_load():
    # The by-kind view is built once by a scan and then kept current by `record` folding each
    # new row in. If the two disagree, the filter holds until the first write and then leaks —
    # a bug that would pass any test that only checks one of the two paths.
    hub = MetadataHub(InProcessBackend())
    hub.record(_feedback("aggregate", fingerprint=hardware.fingerprint(), ms=3.0))
    materialized = hub.op_stats_by_kind()  # forces the one-time load
    assert len(materialized.get("aggregate", [])) == 1

    hub.record(_feedback("aggregate", fingerprint="deadbeef0000", ms=900.0, op_id=1))
    assert len(hub.op_stats_by_kind().get("aggregate", [])) == 1, (
        "a foreign row folded in after the view was materialized must be filtered too"
    )
    hub.record(_feedback("aggregate", fingerprint=hardware.fingerprint(), ms=4.0, op_id=2))
    assert len(hub.op_stats_by_kind().get("aggregate", [])) == 2


def test_feedback_carries_the_measuring_machine():
    fields = OperatorFeedback.__dataclass_fields__
    assert "hw_fingerprint" in fields
    # Defaults to *this* machine, because a feedback row is constructed where the measurement
    # was taken: a caller who does not name a machine is, by construction, saying "here". The
    # exception is a distributed worker's row, which `dist` stamps before it travels.
    built = OperatorFeedback(
        op_id=0,
        kind="scan",
        n_actual=1,
        t_op_ms=1.0,
        m_peak_bytes=1,
        selectivity=1.0,
        batch_size=1,
    )
    assert built.hw_fingerprint == hardware.fingerprint()


def test_a_worker_stamps_its_own_fingerprint_before_shipping():
    # The driver records a worker's metrics, so without a stamp taken at the worker the rows
    # would inherit the *driver's* fingerprint. On a small head node driving large workers
    # that means learning the head node's coefficients for work that never ran there.
    import json

    from batcher.dist.executors.ray_runtime.metering import _stamped_with_this_worker

    doc = json.dumps({"ops": [{"op_id": 0, "kind": "scan"}, {"op_id": 1, "kind": "filter"}]})
    stamped = json.loads(_stamped_with_this_worker(doc))
    assert all(op["hw_fingerprint"] == hardware.fingerprint() for op in stamped["ops"])

    # Best-effort: tagging must never cost a worker its whole contribution.
    assert _stamped_with_this_worker("") == ""
    assert _stamped_with_this_worker("{not json") == "{not json"


def test_unlike_machines_do_not_share_a_fingerprint():
    # The property the whole scheme rests on. If two genuinely different machines collided
    # here, scoping would be decoration and the blend would continue silently.
    small = hardware.HardwareProfile(
        logical_cpus=4,
        physical_cores=2,
        memory_bytes=8 << 30,
        caches={"l2": 512 << 10, "l3": 4 << 20},
        vendor="ARM",
        model="Neoverse-N1",
        simd_bits=128,
        storage_class="network",
    )
    large = dataclasses.replace(
        small,
        logical_cpus=96,
        physical_cores=48,
        memory_bytes=768 << 30,
        caches={"l2": 1 << 20, "l3": 64 << 20},
        vendor="AuthenticAMD",
        model="EPYC 9004",
        simd_bits=512,
        storage_class="nvme",
    )
    assert small.fingerprint() != large.fingerprint()
    assert scoped("ns") != f"ns@{large.fingerprint()}"


def test_a_mixed_cluster_is_detectable_from_the_driver(monkeypatch):
    # A driver cannot see what its workers are. That matters because it is the usual
    # explanation for a learned model that will not converge: an autoscaling group quietly
    # substituting a newer instance generation makes every node's history half about a machine
    # it is not, and nothing about the symptom says so.
    from batcher.dist.executors.ray_runtime import hardware_probe as probe

    big = {"fingerprint": "aaaaaaaaaaaa", "caches": {"l3": 64 << 20}}
    small = {"fingerprint": "bbbbbbbbbbbb", "caches": {"l3": 8 << 20}}

    monkeypatch.setattr(probe, "cluster_hardware_profiles", lambda: (big, small))
    assert probe.cluster_is_heterogeneous() is True
    # The *smallest* cache binds: a broadcast table sized to the big node's cache spills out
    # of the small node's, and the plan does not choose which node it lands on.
    assert probe.cluster_l3_cache_bytes() == 8 << 20

    monkeypatch.setattr(probe, "cluster_hardware_profiles", lambda: (big, dict(big)))
    assert probe.cluster_is_heterogeneous() is False
    assert probe.cluster_l3_cache_bytes() == 64 << 20


def test_an_unprobeable_cluster_reports_unknown_rather_than_a_guess(monkeypatch):
    # Every failure here — Ray down, the probe unschedulable, a worker past the timeout — must
    # leave planning exactly as it was. A driver-local reading substituted for a worker's is
    # the specific wrong answer: a head node is routinely a different machine from the fleet.
    from batcher.dist.executors.ray_runtime import hardware_probe as probe

    monkeypatch.setattr(probe, "cluster_hardware_profiles", tuple)
    assert probe.cluster_l3_cache_bytes() == 0
    assert probe.cluster_is_heterogeneous() is False

    # A node shape whose cache is undetectable must be dropped, not allowed to drag the
    # minimum to zero and disable the broadcast threshold cluster-wide.
    monkeypatch.setattr(
        probe,
        "cluster_hardware_profiles",
        lambda: (
            {"fingerprint": "a", "caches": {}},
            {"fingerprint": "b", "caches": {"l3": 1 << 20}},
        ),
    )
    assert probe.cluster_l3_cache_bytes() == 1 << 20
