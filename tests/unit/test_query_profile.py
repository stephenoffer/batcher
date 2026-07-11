"""`plan.profile` — the planned↔measured join, IR walk, and rendering.

The correctness spine is the `op_id`: the pre-order IR walk MUST reproduce the same
ordering Kyber's `annotate_ops` and the engine's `IdGen` use, or measured metrics get
attributed to the wrong operator. These are pure-Python tests (no native engine).
"""

from __future__ import annotations

import json

import pytest

from batcher.plan.logical import Filter, Scan
from batcher.plan.profile import OpProfile, QueryProfile, build_op_profiles
from batcher.plan.profile.collect import _walk_ir
from batcher.plan.schema import SchemaRef
from batcher.plan.visitor import walk

pytestmark = pytest.mark.unit


def _scan(*names: str) -> Scan:
    import pyarrow as pa

    schema = SchemaRef.from_arrow(pa.schema([(n, pa.int64()) for n in names]))
    return Scan(source_id=0, schema=schema)


def test_ir_walk_matches_logical_walk_order():
    # A plan with an expression-bearing node (Filter) whose predicate IR carries an
    # "op" tag (`gt`) — the walk must NOT descend into it, or op_ids shift.
    from batcher.plan.expr_ir import col

    plan = Filter(_scan("a", "b"), col("a") > 1)
    ir = plan.to_ir()
    walked = [n["op"] for _depth, n in _walk_ir(ir)]
    expected_count = len(list(walk(plan)))  # logical nodes only
    assert walked == ["filter", "scan"]
    assert len(walked) == expected_count  # the predicate's `gt` is not a plan node


def test_build_op_profiles_joins_planned_and_measured_by_op_id():
    ir = {"op": "filter", "input": {"op": "scan", "source_id": 0}, "predicate": {"e": "col"}}
    metrics = [
        {
            "op_id": 0,
            "kind": "filter",
            "rows_in": 100,
            "rows_out": 40,
            "elapsed_ns": 2_000_000,
            "peak_bytes": 4096,
            "backend": "jit",
            "cpu_ns": 1_000_000,
            "threads": 1,
        },
        {
            "op_id": 1,
            "kind": "scan",
            "rows_in": 100,
            "rows_out": 100,
            "elapsed_ns": 500_000,
            "peak_bytes": 8192,
            "backend": "interp",
        },
    ]
    ops = build_op_profiles(ir, (), metrics)
    assert [o.op_id for o in ops] == [0, 1]
    assert [o.kind for o in ops] == ["filter", "scan"]
    assert [o.depth for o in ops] == [0, 1]
    flt = ops[0]
    assert flt.measured and flt.rows_out == 40 and flt.elapsed_ms == 2.0
    assert flt.selectivity == 0.4


def test_planned_only_profile_has_no_measured_ops():
    ir = {"op": "scan", "source_id": 0}
    ops = build_op_profiles(ir, (), None)
    assert len(ops) == 1
    assert not ops[0].measured


def test_est_error_is_actual_over_estimate():
    o = OpProfile(op_id=0, kind="filter", depth=0, est_rows=10.0, measured=True, rows_out=25)
    assert o.est_error == 2.5
    # Unknown estimate → nan (not surfaced).
    import math

    assert math.isnan(OpProfile(op_id=0, kind="x", depth=0, measured=True, rows_out=5).est_error)


def test_query_profile_to_dict_round_trips_through_json():
    ops = (
        OpProfile(
            op_id=0,
            kind="filter",
            depth=0,
            est_rows=10.0,
            measured=True,
            rows_in=100,
            rows_out=40,
            elapsed_ms=2.0,
            backend="jit",
        ),
    )
    profile = QueryProfile(ops=ops, total_ms=5.0, rows=40, measured=True, query_id="q1")
    doc = json.loads(json.dumps(profile.to_dict()))
    assert doc["query_id"] == "q1"
    assert doc["rows"] == 40
    assert doc["ops"][0]["est_error"] == 4.0
    assert doc["ops"][0]["measured"] is True


def test_render_analyze_shows_actual_planned_shows_estimate():
    ops = (
        OpProfile(
            op_id=0,
            kind="filter",
            depth=0,
            est_rows=10.0,
            provenance="exact",
            measured=True,
            rows_in=100,
            rows_out=40,
            elapsed_ms=2.0,
            backend="jit",
        ),
    )
    profile = QueryProfile(ops=ops, total_ms=2.0, rows=40, measured=True)
    analyzed = profile.render(analyze=True)
    assert "actual=40" in analyzed and "bottleneck" in analyzed
    planned = profile.render(analyze=False)
    assert "est≈10" in planned and "actual" not in planned


def test_merge_metric_ops_sums_spill_bytes_across_workers():
    # Each distributed worker spills its own share, so the cluster-wide spill volume is the
    # sum (unlike peak bytes, a concurrent max). Two workers, same op_id.
    from batcher.plan.profile.collect import merge_metric_ops

    w0 = [{"op_id": 0, "kind": "aggregate", "spilled": True, "spill_bytes": 1000}]
    w1 = [{"op_id": 0, "kind": "aggregate", "spilled": True, "spill_bytes": 2500}]
    merged = merge_metric_ops([w0, w1])
    assert merged[0]["spill_bytes"] == 3500
    assert merged[0]["spilled"] is True


def test_render_shows_spill_volume_and_rss():
    # explain(analyze) surfaces the measured spill magnitude and RSS high-water, not just [spill].
    ops = (
        OpProfile(
            op_id=0,
            kind="aggregate",
            depth=0,
            est_rows=100.0,
            measured=True,
            rows_in=1_000_000,
            rows_out=4,
            elapsed_ms=5.0,
            spilled=True,
            spill_bytes=2_000_000_000,
            peak_rss_bytes=1_500_000_000,
        ),
    )
    out = QueryProfile(ops=ops, total_ms=5.0, rows=4, measured=True).render(analyze=True)
    assert "spill" in out and "GB" in out  # magnitude shown, not just the bare tag
    assert "rss+" in out


def test_summary_aggregates_spill_and_classifies_by_cpu_util():
    # A low measured CPU utilization classifies the bottleneck as I/O/launch-bound even
    # for a non-scan op, and the summary reports total spill volume.
    ops = (
        OpProfile(
            op_id=0,
            kind="aggregate",
            depth=0,
            measured=True,
            rows_in=1_000,
            rows_out=4,
            elapsed_ms=10.0,
            spilled=True,
            spill_bytes=1_500_000_000,
            peak_rss_bytes=800_000_000,
            cpu_util=0.2,  # I/O/launch-bound despite being an aggregate
        ),
    )
    p = QueryProfile(ops=ops, total_ms=10.0, rows=4, measured=True)
    assert p.total_spill_bytes == 1_500_000_000
    assert p.peak_rss_bytes == 800_000_000
    summary = p.bottleneck_summary()
    assert "I/O/launch-bound" in summary
    assert "SPILLED" in summary and "GB" in summary


def test_utilization_summary_grades_against_saturation_target():
    def _op(op_id, cpu_util, elapsed_ms):
        return OpProfile(
            op_id=op_id,
            kind="filter",
            depth=0,
            measured=True,
            rows_in=1_000,
            rows_out=1_000,
            elapsed_ms=elapsed_ms,
            cpu_util=cpu_util,
        )

    # Wall-time-weighted: a long 0.95-util op dominates a brief 0.1-util one → "saturated".
    hot = QueryProfile(
        ops=(_op(0, 0.95, 100.0), _op(1, 0.1, 1.0)), total_ms=101.0, rows=1000, measured=True
    )
    assert hot.cpu_utilization_overall() == pytest.approx((0.95 * 100 + 0.1 * 1) / 101.0)
    hot_summary = hot.utilization_summary()
    assert "cpu utilization: 94%" in hot_summary and "saturated" in hot_summary
    assert hot_summary in hot.render(analyze=True)

    # With a known budget, peak memory is reported as a fraction of it (the >80% memory target).
    with_mem = QueryProfile(
        ops=(
            OpProfile(
                op_id=0,
                kind="aggregate",
                depth=0,
                measured=True,
                elapsed_ms=10.0,
                cpu_util=0.95,
                peak_rss_bytes=800,
            ),
        ),
        total_ms=10.0,
        rows=1,
        measured=True,
        memory_budget_bytes=1000,
    )
    assert "peak memory 800B (80% of budget, target >80%)" in with_mem.utilization_summary()

    # A uniformly I/O-bound run is flagged as CPU-idle (the CPU is not the limit — no false alarm).
    cold = QueryProfile(ops=(_op(0, 0.2, 50.0),), total_ms=50.0, rows=1000, measured=True)
    assert "CPU idle" in cold.utilization_summary()

    # No CPU time measured → no line (empty), and JSON carries the numeric fields.
    bare = QueryProfile(ops=(), total_ms=0.0, rows=0, measured=True)
    assert bare.utilization_summary() == ""
    assert "cpu_utilization" in hot.to_dict() and hot.to_dict()["cpu_utilization"] > 0.9
