"""Findings about the Python-UDF stages of an ML pipeline.

These read the per-stage measurements the orchestrator records for a `map_batches` chain
(`plan.profile.StageRecorder`) rather than the engine's operator metrics, because that is
the only place a batch-inference pipeline's numbers exist. They are separated from the
relational rule families for the same reason: a relational finding says "the plan is wrong",
these say "the *shape of the pipeline* is wrong", and the two call for different fixes.

Each rule here corresponds to a failure the Ray field guides diagnose by hand, from
`ds.stats()` output the user has to read and interpret. The point of having them as rules is
that nobody has to read anything: the run says what went wrong.
"""

from __future__ import annotations

from typing import Any

from batcher.observe.insights.kinds import (
    _GPU_STARVED_MIN_MS,
    _GPU_STARVED_RATIO,
    _PER_ROW_MIN_ROWS,
    _PER_ROW_SHARE,
    _STAGE_EXPLOSION_MIN_ROWS,
    _STAGE_EXPLOSION_RATIO,
    _UDF_DOMINATES_MIN_MS,
    _UDF_DOMINATES_SHARE,
    Insight,
    count,
)

__all__ = ["gpu_starved", "per_row_map", "row_exploding_stage", "udf_dominates"]

#: How a UDF stage is named in both op-id spaces — the logical tree uses the node's class
#: name, the recorder uses the IR tag. Matching either keeps the rules working whichever
#: profile shape they are handed.
_UDF_KINDS = frozenset({"MapBatches", "map_batches", "MapRows"})
#: A scan is not "relational work the optimizer could have done differently" — it is the
#: read every plan has. Spelled both ways for the same reason as `_UDF_KINDS`: the logical
#: tree names operators by class, the engine profile by IR tag.
_SCAN_KINDS = frozenset({"Scan", "scan"})


def _udf_stages(ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The measured `map_batches` stages of a profile, in plan order."""
    return [op for op in ops if op.get("kind") in _UDF_KINDS]


def gpu_starved(
    _profile: dict[str, Any], ops: list[dict[str, Any]], _total_ms: float
) -> list[Insight]:
    """A GPU stage waiting on the CPU stages that feed it.

    The field guides call CPU preprocessing starving the GPU the single most common cause of
    low GPU utilization, and it is invisible in a plan: the pipeline is *correct*, the device
    is simply idle. It shows up only as the ratio between what the CPU stages cost and what
    the GPU stage cost, which is why it needs measured per-stage time to detect at all.

    Batcher already overlaps the two (the CPU stage prepares morsel *k+1* while the GPU runs
    *k*), so this fires when overlapping is not enough — the CPU work genuinely exceeds the
    GPU work, and the answer is more CPU per GPU rather than more scheduling.
    """
    stages = _udf_stages(ops)
    gpu = [op for op in stages if op.get("backend") == "gpu"]
    cpu = [op for op in stages if op.get("backend") != "gpu"]
    if not gpu or not cpu:
        return []
    gpu_ms = sum(float(op.get("elapsed_ms", 0.0)) for op in gpu)
    cpu_ms = sum(float(op.get("elapsed_ms", 0.0)) for op in cpu)
    # The size floor is on the pipeline, not the GPU stage — see `_GPU_STARVED_MIN_MS`.
    if cpu_ms + gpu_ms < _GPU_STARVED_MIN_MS or cpu_ms < gpu_ms * _GPU_STARVED_RATIO:
        return []
    return [
        Insight(
            severity="warning",
            rule="gpu-starved",
            title=f"CPU stages cost {cpu_ms / max(gpu_ms, 1e-9):.1f}x the GPU stage they feed",
            evidence=(
                f"{len(cpu)} CPU stage(s) spent {cpu_ms:.0f} ms producing input for "
                f"{len(gpu)} GPU stage(s) that spent {gpu_ms:.0f} ms consuming it. The stages "
                f"already overlap, so the device still waits: there is more CPU work than "
                f"there is GPU work to hide it behind."
            ),
            action=(
                "Give the pipeline more CPU per GPU — a larger CPU allocation, or a cluster "
                "shape with more CPU workers per GPU worker. If the CPU stage is decoding, "
                "check it is not decoding columns the GPU stage never reads."
            ),
            detail={"cpu_ms": round(cpu_ms, 1), "gpu_ms": round(gpu_ms, 1)},
        )
    ]


def udf_dominates(
    _profile: dict[str, Any], ops: list[dict[str, Any]], _total_ms: float
) -> list[Insight]:
    """Python UDF stages owning most of a plan that also has relational work.

    A `map_batches` is opaque: the optimizer cannot push a predicate through it, cannot fuse
    across it, and cannot estimate past it. When the UDFs own most of the runtime *and* there
    are relational operators in the same plan, some of that work is usually expressible as an
    expression — which runs in Rust and, more importantly, stops being a wall in the plan.

    A pipeline that is *only* UDFs (the pure batch-inference shape) is left alone: there is
    nothing to push and no relational half to speed up, so the advice would be noise.
    """
    stages = _udf_stages(ops)
    relational = [op for op in ops if op.get("kind") not in _UDF_KINDS | _SCAN_KINDS]
    if not stages or not relational:
        return []
    udf_ms = sum(float(op.get("elapsed_ms", 0.0)) for op in stages)
    measured_ms = sum(float(op.get("elapsed_ms", 0.0)) for op in ops)
    if udf_ms < _UDF_DOMINATES_MIN_MS or udf_ms < measured_ms * _UDF_DOMINATES_SHARE:
        return []
    return [
        Insight(
            severity="info",
            rule="udf-dominates",
            title=f"Python stages own {udf_ms / max(measured_ms, 1e-9):.0%} of the measured time",
            evidence=(
                f"{len(stages)} map_batches stage(s) spent {udf_ms:.0f} ms of {measured_ms:.0f} ms "
                f"measured, alongside {len(relational)} relational operator(s). The optimizer "
                f"cannot see through a Python callback, so nothing moves across those stages."
            ),
            action=(
                "Where a stage is arithmetic, string work, or a comparison, express it with "
                "`bt.col(...)` instead: it runs in Rust, and the filter above it becomes "
                "pushable. Keep `map_batches` for the model call it exists for."
            ),
            detail={"udf_ms": round(udf_ms, 1), "measured_ms": round(measured_ms, 1)},
        )
    ]


def row_exploding_stage(
    _profile: dict[str, Any], ops: list[dict[str, Any]], _total_ms: float
) -> list[Insight]:
    """A UDF stage emitting far more rows than it consumed.

    One-to-many stages (chunking a document, sampling frames from a video, augmenting an
    image) multiply everything downstream: the row count, the memory, and the cost of every
    later stage. That is often intended — but it is the one shape where a batch size chosen
    for the *input* is the wrong size for the output, and it is worth naming.
    """
    found: list[Insight] = []
    for op in _udf_stages(ops):
        rows_in = int(op.get("rows_in", 0) or 0)
        rows_out = int(op.get("rows_out", 0) or 0)
        if rows_in < _STAGE_EXPLOSION_MIN_ROWS or rows_out < rows_in * _STAGE_EXPLOSION_RATIO:
            continue
        found.append(
            Insight(
                severity="info",
                rule="row-exploding-stage",
                title=f"A stage turned {count(rows_in)} rows into {count(rows_out)}",
                evidence=(
                    f"Step {op.get('op_id')} consumed {count(rows_in)} rows and emitted "
                    f"{count(rows_out)} — {rows_out / max(rows_in, 1):.1f}x. Every stage after "
                    f"it processes the multiplied row count."
                ),
                action=(
                    "If a stage below this one filters, move it above the fan-out so the "
                    "multiplication happens on fewer rows. If the expansion factor is large, "
                    "a smaller input batch keeps the stage's transient output bounded."
                ),
                detail={"op_id": op.get("op_id"), "rows_in": rows_in, "rows_out": rows_out},
            )
        )
    return found


def per_row_map(
    _profile: dict[str, Any], ops: list[dict[str, Any]], _total_ms: float
) -> list[Insight]:
    """A per-row `map` carrying a large share of the run.

    `map` calls a Python function once per row; `map_batches` calls one per Arrow batch. For
    anything expressible over columns the field guides measure the gap at 10-100x, and it is
    the single most common shape in their catalog. The two are the same operator to the
    engine — `map` lowers to `map_batches` over a row loop — so only the profile can tell
    them apart, and only once the run has happened.

    Per-row is sometimes exactly right: genuinely row-shaped work, or an async per-row API
    call whose cost is the network. So this stays quiet until the stage is both large enough
    to matter and actually owns a real share of the time.
    """
    found: list[Insight] = []
    measured_ms = sum(float(op.get("elapsed_ms", 0.0)) for op in ops) or 1.0
    for op in ops:
        if op.get("kind") != "MapRows":
            continue
        rows = int(op.get("rows_in", 0) or 0)
        elapsed = float(op.get("elapsed_ms", 0.0))
        if rows < _PER_ROW_MIN_ROWS or elapsed < measured_ms * _PER_ROW_SHARE:
            continue
        found.append(
            Insight(
                severity="warning",
                rule="per-row-map",
                title=f"A per-row map over {count(rows)} rows owns {elapsed / measured_ms:.0%}",
                evidence=(
                    f"Step {op.get('op_id')} called a Python function once per row for "
                    f"{count(rows)} rows, taking {elapsed:.0f} ms of {measured_ms:.0f} ms "
                    f"measured. A batch call would run once per Arrow batch instead."
                ),
                action=(
                    "If the work is expressible over columns, use `bt.col(...)` expressions "
                    "(Rust, no Python per row) or `map_batches` (one call per batch). Keep "
                    "`map` for genuinely row-shaped work or an async per-row API call."
                ),
                detail={"op_id": op.get("op_id"), "rows": rows, "elapsed_ms": round(elapsed, 1)},
            )
        )
    return found
