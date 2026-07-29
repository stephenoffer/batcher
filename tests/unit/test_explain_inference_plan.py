"""`explain()` on a batch-inference / `map_batches` pipeline renders instead of crashing.

A `map_batches` node is executed in Python and deliberately has no engine IR (`to_ir()`
raises), so the planned `explain()` path used to lower the plan and crash with
`NotImplementedError` — making the plan of an inference pipeline impossible to inspect,
which is step one of debugging that workload class. The planned view now falls back to the
un-lowered logical tree, and the analyzed view (`analyze=True`/`stats()`) measures each
stage in the orchestrator rather than refusing for want of engine metrics.
"""

from __future__ import annotations

import json

import pytest

import batcher as bt

pytestmark = pytest.mark.unit


def _inference_plan() -> bt.Dataset:
    """A filter → map_batches (opaque UDF) pipeline, the batch-inference shape."""
    ds = bt.from_pydict({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
    return ds.filter(bt.col("a") > 1).map_batches(lambda batch: batch)


def _relational_plan() -> bt.Dataset:
    """A pure-relational filter + group_by/agg pipeline (fully lowerable)."""
    ds = bt.from_pydict({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
    return ds.filter(bt.col("a") > 1).group_by("a").agg(bt.col("b").sum())


def test_explain_map_batches_renders_tree() -> None:
    """A `map_batches` plan's planned `explain()` returns a tree naming `MapBatches`.

    Before the fix this raised `NotImplementedError` (map_batches has no engine IR); the
    regression guard below (`_raises_before_fix`) documents that prior behavior.
    """
    text = _inference_plan().explain()
    assert isinstance(text, str)
    assert text.strip()
    assert "MapBatches" in text
    # The un-lowered fallback is honestly labelled, not silently passed off as an
    # optimized plan.
    assert "un-lowered" in text


def test_explain_map_batches_json_document() -> None:
    """`explain(format="json")` on a UDF plan returns a JSON document, not a crash."""
    doc = _inference_plan().explain(format="json")
    parsed = json.loads(doc)
    assert isinstance(parsed, dict)
    kinds = [op["kind"] for op in parsed["ops"]]
    assert "MapBatches" in kinds


def test_explain_analyze_measures_every_stage_of_a_udf_plan() -> None:
    """`explain(analyze=True)` on a UDF plan measures each stage instead of refusing.

    The engine emits no `ExecMetrics` for a Python UDF, which used to mean the analyzed
    path raised `BackendError` — leaving the batch-inference shape, the one the ML surface
    exists for, with no answer to "which stage is the bottleneck". The orchestrator now
    measures the stages itself and renders them against the logical tree.
    """
    text = _inference_plan().explain(analyze=True)
    assert "MapBatches" in text
    # `actual=` is the measured-row marker the analyzed renderer adds; its presence is what
    # separates a real measurement from the planned tree being passed off as one.
    assert "actual=" in text


def test_udf_stage_timings_attribute_to_the_right_stage() -> None:
    """A slow stage's time lands on that stage, not smeared across the tree.

    The whole point of a per-stage profile is telling a slow stage from a fast one, so a
    profile that merely *has* numbers is not enough — they have to be on the right rows.
    """
    import time

    ds = bt.from_pydict({"a": list(range(512))})
    stats = ds.map_batches(lambda b: b).map_batches(_slow(time)).stats()
    stages = [op for op in stats.ops if op.kind == "MapBatches"]
    assert len(stages) == 2, "both UDF stages must be measured"
    slow_ms = max(op.elapsed_ms for op in stages)
    fast_ms = min(op.elapsed_ms for op in stages)
    assert slow_ms > fast_ms * 5, f"the sleeping stage must dominate ({slow_ms=}, {fast_ms=})"
    assert all(op.rows_out == 512 for op in stages)


def _slow(time_module):
    """A UDF that costs a measurable, unmistakable amount of wall time."""

    def stage(batch):
        time_module.sleep(0.02)
        return batch

    return stage


def test_relational_explain_is_byte_identical() -> None:
    """A pure-relational plan's `explain()` output is unchanged by the UDF fallback.

    The fix adds a branch that only triggers on a `map_batches` plan; a lowerable plan must
    render exactly as before. The expected string is the captured pre-change output.
    """
    expected = (
        "aggregate                       est≈1 (default)\n"
        "  filter                        est≈1 (default)\n"
        "    scan                        est≈4 (exact)"
    )
    assert _relational_plan().explain() == expected


def test_a_stage_that_could_not_count_its_input_reads_it_off_the_tree() -> None:
    """The streaming path meters a stage by wrapping its output generator, which never sees
    the input — so it reports `rows_in=0`. Rendering that verbatim reads as "this stage
    consumed nothing" rather than "this seam could not observe it", and in a linear chain
    the answer is sitting on the node directly beneath."""
    from batcher.api.terminal.profile import _logical_op_profiles

    plan = _inference_plan()._plan
    # Shaped like the streaming recorder's output: rows_out known, rows_in unobservable.
    metrics = [
        {"op_id": 0, "kind": "map_batches", "rows_in": 0, "rows_out": 3, "elapsed_ns": 1000},
        {"op_id": 1, "kind": "filter", "rows_in": 0, "rows_out": 3, "elapsed_ns": 500},
    ]
    ops = _logical_op_profiles(plan, metrics)
    assert ops[0].rows_in == 3, "the stage's input is the row count of the node below it"


def test_an_unmeasured_child_does_not_invent_an_input_count() -> None:
    from batcher.api.terminal.profile import _logical_op_profiles

    plan = _inference_plan()._plan
    ops = _logical_op_profiles(
        plan, [{"op_id": 0, "kind": "map_batches", "rows_in": 0, "rows_out": 3, "elapsed_ns": 1}]
    )
    assert ops[0].rows_in == 0


def test_a_streamed_stage_is_timed_on_its_own_work_not_its_upstreams_wait() -> None:
    """The stage-overlapped path runs each stage on its own thread behind a queue. Timing a
    stage's *output generator* would charge it for waiting on a slower upstream — an
    unbounded distortion that makes the last stage always look dominant, and that would
    invert `gpu-starved`, whose whole job is comparing a GPU stage against its feeders.

    Here the CPU stage sleeps 8x longer per batch than the GPU stage, so it must read as the
    expensive one even though the GPU stage is downstream of it."""
    import time

    import pyarrow as pa

    class Decode:
        def __call__(self, batch):
            time.sleep(0.004)
            return batch

    class Model:
        def __call__(self, batch):
            time.sleep(0.0005)
            return batch

    ds = (
        bt.from_arrow(pa.table({"x": list(range(2000))}))
        .ml.map_batches(Decode, batch_size=250)
        .ml.map_batches(Model, num_gpus=1, batch_size=250)
    )
    ds.collect()  # warm up: the first num_gpus>0 run pays one-time device detection
    stats = ds.stats()
    stages = {op.backend: op for op in stats.ops if op.kind == "MapBatches"}
    assert set(stages) == {"gpu", ""}, "both stages must be measured and distinguishable"
    assert stages[""].elapsed_ms > stages["gpu"].elapsed_ms * 3, (
        f"the slow CPU stage must own the time, not the GPU stage it feeds ({stages=})"
    )
    # Input rows are observed directly on this path, not inferred from the tree.
    assert all(op.rows_in == 2000 for op in stages.values())


def test_a_per_row_map_is_named_as_such_in_the_tree() -> None:
    """`ds.map` lowers to `map_batches` over a row loop, so the plan tree shows one node for
    both. The measured tree distinguishes them, because the cost difference is 10-100x."""
    ds = bt.from_pydict({"x": list(range(256))})
    kinds = [op.kind for op in ds.ml.map(lambda row: {"x": row["x"] + 1}).stats().ops]
    assert "MapRows" in kinds
    assert "MapBatches" not in kinds


def test_a_batch_map_is_still_named_map_batches() -> None:
    ds = bt.from_pydict({"x": list(range(256))})
    kinds = [op.kind for op in ds.map_batches(lambda b: b).stats().ops]
    assert "MapBatches" in kinds
    assert "MapRows" not in kinds
