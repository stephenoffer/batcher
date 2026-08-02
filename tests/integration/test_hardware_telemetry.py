"""Per-operator hardware telemetry reaches the control plane, and never lies when it can't.

The engine measures what each operator cost the machine — pages faulted, CPU evictions, bytes
that actually reached the disk — and rides it back beside the result batches. Two properties
matter and are tested here.

First, the counters must actually arrive: the wire contract runs Rust `serde` to a JSON
document to `OperatorFeedback` and `OpProfile`, and a rename on either side silently delivers
zeros forever, because nothing raises when a key is missing from a `dict.get`.

Second, an unmeasurable counter must read as `0`, and every consumer must treat `0` as
"unmeasured" rather than "none". A fabricated plausible default here would propagate into a
learned cost model that then describes a machine nobody ran on.
"""

from __future__ import annotations

import json

import pytest

import batcher as bt
from batcher._internal import hardware
from batcher.plan.feedback import (
    CONTENDED_PREEMPTIONS_PER_CORE_SECOND,
    OperatorFeedback,
    preemption_rate,
)
from batcher.plan.profile.types import OpProfile

pytestmark = pytest.mark.integration

# The hardware counters the engine flattens into its metrics document, and that every layer
# above must carry unchanged. Named once so a rename breaks this list rather than silently
# zeroing a field nobody notices.
HW_FIELDS = (
    "minor_faults",
    "major_faults",
    "vol_ctx_switches",
    "invol_ctx_switches",
    "io_read_bytes",
    "io_write_bytes",
)


def _analyzed(dataset) -> dict:
    """The machine-readable `EXPLAIN ANALYZE` document for a dataset."""
    return json.loads(dataset.explain(analyze=True, format="json"))


def test_every_operator_carries_the_hardware_fields():
    ds = bt.from_pydict({"k": [i % 101 for i in range(50_000)], "v": list(range(50_000))})
    doc = _analyzed(ds.filter(bt.col("v") > 5).group_by("k").agg(s=bt.col("v").sum()))
    assert doc["ops"], "an analyzed profile must report operators"
    for op in doc["ops"]:
        for field in (*HW_FIELDS, "preemption_rate"):
            assert field in op, f"{op['kind']} is missing {field} — the wire contract drifted"
            assert op[field] >= 0, f"{op['kind']}.{field} must never be negative"


def test_counters_are_measured_on_the_materializing_path():
    # The counters are read from the OS at operator boundaries, which the materializing
    # executor owns. A run that allocates tens of megabytes must fault pages in; if this
    # reports zero the sampling is not wired to the record sites at all.
    import dataclasses

    from batcher.config import active_config, set_config

    rows = 300_000
    ds = bt.from_pydict({"k": [i % 977 for i in range(rows)], "v": list(range(rows))})
    prev = active_config()
    pinned = prev.replace(execution=dataclasses.replace(prev.execution, streaming=False))
    set_config(pinned)
    try:
        doc = _analyzed(ds.group_by("k").agg(s=bt.col("v").sum()))
    finally:
        set_config(prev)
    measured = [o for o in doc["ops"] if o["measured"]]
    assert measured, "the materializing path must measure its operators"
    total_faults = sum(o["minor_faults"] for o in measured)
    switches = sum(o["vol_ctx_switches"] + o["invol_ctx_switches"] for o in measured)
    assert total_faults + switches > 0, (
        "a multi-megabyte run must register page faults or context switches; "
        "all-zero means the OS sampler is not reaching the metric record sites"
    )


def test_streaming_cpu_utilization_is_measured_not_a_constant():
    # The streaming tier is the default path, and it used to report `1 / threads` for every
    # operator of every query — a constant dressed as a measurement, which `explain(analyze)`
    # printed as a verdict and the learned CPU-share model fitted against. The property that
    # catches a return to that is not the absolute value but that it *moves with the work*:
    # a query with twenty times the rows must keep the cores busier.
    def utilization(rows: int) -> float:
        ds = bt.from_pydict({"k": [i % 50_000 for i in range(rows)], "v": list(range(rows))})
        q = ds.filter(bt.col("v") % 7 > 1).group_by("k").agg(s=bt.col("v").sum())
        return json.loads(q.explain(analyze=True, format="json"))["cpu_utilization"]

    small, large = utilization(200_000), utilization(4_000_000)
    assert small > 0.0, "the streaming tier must report a measured CPU figure, not zero"
    assert large > small, (
        f"utilization must rise with real work ({small:.4f} -> {large:.4f}); "
        "a figure that does not move with the workload is a constant, not a measurement"
    )


def test_the_wall_span_is_what_makes_the_streaming_figure_possible():
    from batcher.plan.feedback import cpu_utilization

    # Summed busy time divided by itself is exactly 1/threads, whatever the operator did.
    # That is the shape of the bug: pass the same number twice and every operator on a
    # 16-thread pool reports 6.25%.
    busy_ns = 8_000_000.0
    assert cpu_utilization(busy_ns, busy_ns, 16) == pytest.approx(1 / 16)
    # Given the interval the operator actually occupied, the same inputs describe reality:
    # 8 ms of work spread over a 2 ms window on 16 threads is 25% of the pool.
    assert cpu_utilization(busy_ns, busy_ns, 16, wall_span_ns=2_000_000.0) == pytest.approx(0.25)
    # A tier that tracks no span keeps dividing by its own elapsed time, which is correct
    # there because a materializing operator runs alone and owns its wall interval.
    assert cpu_utilization(1_000.0, 4_000.0, 1, wall_span_ns=0.0) == pytest.approx(0.25)


def test_feedback_carries_the_counters_to_the_metadata_hub():
    # The learned models read feedback rows, not the profile. A field present in `OpProfile`
    # and absent from `OperatorFeedback` is invisible to every adaptive loop.
    fields = OperatorFeedback.__dataclass_fields__
    for field in HW_FIELDS:
        assert field in fields, f"OperatorFeedback must carry {field}"
        assert fields[field].default == 0, f"{field} must default to 0 (unmeasured)"


def test_preemption_rate_normalizes_per_core_second():
    # 1,000 evictions over one second on four cores is 250 per core-second, not 1,000: an
    # operator twice as wide sees twice the raw switches for the same contention.
    assert preemption_rate(1_000, elapsed_ms=1_000, threads=4) == pytest.approx(250.0)
    # Unmeasured inputs yield no signal rather than a divide-by-zero on the query path.
    assert preemption_rate(0, elapsed_ms=1_000, threads=4) == 0.0
    assert preemption_rate(1_000, elapsed_ms=0, threads=4) == 0.0
    assert preemption_rate(1_000, elapsed_ms=1_000, threads=0) == 0.0


def test_a_contended_operator_is_flagged_in_the_rendered_plan():
    # The plan line stays quiet on a healthy operator and speaks up on a contended one, so a
    # reader is not asked to scan a row of zeros on every query to find the one that matters.
    healthy = OpProfile(op_id=0, kind="filter", depth=0, measured=True, elapsed_ms=100.0, threads=4)
    assert healthy.contended is False
    assert healthy.paging is False

    # Enough evictions to clear the threshold across 4 cores for 100 ms of wall time.
    switches = int(CONTENDED_PREEMPTIONS_PER_CORE_SECOND * 0.1 * 4) + 100
    contended = OpProfile(
        op_id=0,
        kind="filter",
        depth=0,
        measured=True,
        elapsed_ms=100.0,
        threads=4,
        invol_ctx_switches=switches,
        major_faults=3,
    )
    assert contended.contended is True
    assert contended.paging is True


def test_the_plan_line_names_paging_and_contention():
    from batcher.plan.profile.types import QueryProfile

    op = OpProfile(
        op_id=0,
        kind="aggregate",
        depth=0,
        measured=True,
        elapsed_ms=100.0,
        threads=4,
        rows_in=10,
        rows_out=10,
        invol_ctx_switches=100_000,
        major_faults=4_096,
        io_read_bytes=1 << 20,
    )
    rendered = QueryProfile(ops=(op,), total_ms=100.0, measured=True).render(analyze=True)
    assert "PAGING" in rendered, "a paging operator must say so — it invalidates every timing"
    assert "contended" in rendered
    assert "disk-read" in rendered


def test_a_healthy_plan_line_stays_uncluttered():
    from batcher.plan.profile.types import QueryProfile

    op = OpProfile(
        op_id=0, kind="filter", depth=0, measured=True, elapsed_ms=5.0, threads=4, rows_out=10
    )
    rendered = QueryProfile(ops=(op,), total_ms=5.0, measured=True).render(analyze=True)
    for noise in ("PAGING", "contended", "disk-read", "disk-write"):
        assert noise not in rendered, f"{noise} must not appear on a healthy operator"


def test_distributed_merge_sums_the_event_counters():
    # Faults and switches are counts of things that happened, so the cluster total is the sum
    # across workers — unlike peak bytes, which is a concurrent high-water mark.
    from batcher.plan.profile.collect import merge_metric_ops

    worker = {
        "op_id": 0,
        "kind": "aggregate",
        "rows_in": 10,
        "rows_out": 5,
        "elapsed_ns": 1_000,
        "cpu_ns": 900,
        "peak_bytes": 4096,
        "result_bytes": 1024,
        "threads": 2,
        "minor_faults": 7,
        "major_faults": 1,
        "vol_ctx_switches": 3,
        "invol_ctx_switches": 2,
        "io_read_bytes": 8192,
        "io_write_bytes": 512,
    }
    merged = merge_metric_ops([[worker], [dict(worker)], [dict(worker)]])
    assert len(merged) == 1
    assert merged[0]["minor_faults"] == 21
    assert merged[0]["major_faults"] == 3
    assert merged[0]["io_read_bytes"] == 3 * 8192
    # Wall time and peak bytes keep their non-additive merges.
    assert merged[0]["elapsed_ns"] == 1_000
    assert merged[0]["peak_bytes"] == 4096


def test_the_profile_names_the_machine_it_was_measured_on():
    from batcher.plan.profile.types import QueryProfile

    # Every timing in a profile is relative to a machine, and a profile read out of a log or
    # compared against one from another node is otherwise unattributed. The same string is the
    # key the engine's learned costs are stored under, so it also answers "would what this run
    # learned apply to that other node?".
    op = OpProfile(op_id=0, kind="scan", depth=0, measured=True, elapsed_ms=5.0, threads=4)
    profile = QueryProfile(ops=(op,), total_ms=5.0, measured=True, rows=1)

    machine = profile.machine
    assert machine and "[" in machine and machine.endswith("]")
    assert hardware.fingerprint() in machine

    rendered = profile.render(analyze=True)
    assert f"machine: {machine}" in rendered
    assert profile.to_dict()["machine"] == machine

    # A planned-only profile prints no measurements, so naming the machine there would be
    # answering a question nobody asked.
    assert "machine:" not in QueryProfile(ops=(op,)).render(analyze=False)

    # And a distributed profile is assembled on the driver while the work ran on the workers.
    # Printing the driver's machine there would attribute every timing above it to hardware
    # that ran none of it — a head node is routinely a different shape from its fleet.
    distributed = QueryProfile(ops=(op,), total_ms=5.0, measured=True, rows=1, distributed=True)
    assert "machine:" not in distributed.render(analyze=True)
    # Still in the JSON, where a consumer can see which node assembled the document.
    assert distributed.to_dict()["machine"] == machine
