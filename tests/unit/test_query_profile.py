"""`plan.profile` — the planned↔measured join, IR walk, and rendering.

The correctness spine is the `op_id`: the pre-order IR walk MUST reproduce the same
ordering Kyber's `_annotate_ops` and the engine's `IdGen` use, or measured metrics get
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
