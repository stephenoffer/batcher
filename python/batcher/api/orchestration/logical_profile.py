"""`OpProfile`s built from the un-lowered LOGICAL plan tree.

A `map_batches` / inference pipeline has no engine IR -- its `to_ir()` deliberately raises,
which is also why `kyber.optimize` refuses it -- so the estimate/measure join that
`plan.profile.build_op_profiles` performs off the lowered IR cannot run for it. Everything
here is the other seam: read the logical tree directly, name each operator by its node
type, and attach whatever Kyber and Core can say about it without a `PhysicalPlan`.

Split out of `api.terminal.profile`, which was within six lines of the 500-line limit and
in a directory at its 12-file cap. It sits beside `sizing` rather than under `terminal`
because what it does is ask Kyber what it knows about a plan, which is the orchestrator's
side of the boundary; the terminal's profile consumes the answer. The names stay importable
from `api.terminal.profile` too, since `event_log` and
`tests/unit/test_explain_inference_plan.py` reach them by that path.
"""

from __future__ import annotations

from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan
from batcher.plan.profile import OpProfile

__all__ = ["_logical_estimates", "_logical_op_profiles", "_rows_in"]


def _logical_estimates(plan: LogicalPlan, sources: list[Source]) -> dict[int, float]:
    """Kyber's row estimate per logical node id, for the operators it can estimate.

    A UDF plan cannot be lowered -- `kyber.optimize` raises `NotImplementedError` on a
    `MapBatches`, which is what keeps the whole optimizer off this path -- but *cardinality
    estimation* needs no lowering. `StatsEstimator.estimate` walks the logical tree, so the
    `Scan` beneath a UDF reports the exact row count it reports on any other plan.

    `MapBatches` is deliberately left out. The estimator will answer for it, by applying a
    default selectivity to an opaque Python function, and that number is a guess rather
    than an estimate: publishing it would make `est_error` compare a measurement against
    one. Declining it is the same choice the interpreter/JIT parity rule makes elsewhere --
    fall back visibly rather than answer wrongly. Every operator Kyber genuinely knows
    something about now carries what it knows, and the one it does not carries nothing.

    Best-effort: an estimator that cannot be built leaves every row planned-only, exactly
    as before, rather than failing a query whose result is fine.
    """
    from batcher.plan.logical import MapBatches
    from batcher.plan.profile import logical_preorder

    try:
        from batcher.kyber.stats.estimator import StatsEstimator

        estimator = StatsEstimator(sources)
        out: dict[int, float] = {}
        for op_id, (_depth, node) in enumerate(logical_preorder(plan)):
            if isinstance(node, MapBatches):
                continue
            rows = getattr(estimator.estimate(node), "rows", None)
            if rows is not None:
                out[op_id] = float(rows)
        return out
    except Exception:  # pragma: no cover - an estimate that cannot be made is not a failure
        from batcher._internal.logging import note_suppressed

        note_suppressed("udf-logical-estimates")
        return {}


def _logical_op_profiles(
    plan: LogicalPlan,
    metric_ops: list[dict] | None = None,
    est_rows: dict[int, float] | None = None,
) -> list[OpProfile]:
    """`OpProfile`s from the un-lowered LOGICAL plan tree, pre-order.

    The seam a UDF plan takes: a `map_batches`/inference pipeline has no engine IR (its
    `to_ir()` deliberately raises), so the estimate/measure join `build_op_profiles`
    performs off the lowered IR cannot run. This walks the logical tree via
    `logical_preorder`, naming each operator by its node type, so `explain()` renders a
    readable operator tree — `MapBatches` included — instead of crashing.

    `metric_ops` are the `StageRecorder`'s measurements for the same tree, numbered by the
    same walk, so `stats()` on an ML pipeline shows measured rows and time per stage rather
    than refusing. `None` leaves every row planned-only, which is `explain()` without
    `analyze`.

    `est_rows` is Kyber's estimate per op id, from `_logical_estimates`, for the operators
    it can estimate. Absent for `MapBatches` by design; see that function.
    """
    from batcher.plan.profile import logical_preorder

    measured = {int(m.get("op_id", -1)): m for m in (metric_ops or [])}
    # `OpProfile.est_rows` uses NaN, not None, for "not estimated" -- `to_dict` renders NaN
    # as JSON null. Passing None here is a TypeError in `math.isnan`, which is what a
    # `MapBatches` (deliberately unestimated) produced on the first attempt.
    estimates = est_rows or {}
    unknown = float("nan")
    nodes = list(logical_preorder(plan))
    out: list[OpProfile] = []
    for op_id, (depth, node) in enumerate(nodes):
        m = measured.get(op_id)
        estimate = estimates.get(op_id, unknown)
        # Prefer the measured operator name, as `build_op_profiles` does: it is the only
        # thing that can tell a per-row `map` from a vectorized `map_batches`, which are the
        # same node type but 10-100x apart in cost.
        kind = str(m.get("kind")) if m and m.get("kind") else type(node).__name__
        if m is None:
            out.append(OpProfile(op_id=op_id, kind=kind, depth=depth, est_rows=estimate))
            continue
        out.append(
            OpProfile(
                op_id=op_id,
                kind=kind,
                depth=depth,
                measured=True,
                est_rows=estimate,
                rows_in=_rows_in(m, op_id, nodes, measured),
                rows_out=int(m.get("rows_out", 0)),
                elapsed_ms=float(m.get("elapsed_ns", 0)) / 1e6,
                result_bytes=int(m.get("result_bytes", 0)),
                threads=int(m.get("threads", 0)),
                backend=str(m.get("backend", "")),
            )
        )
    return out


def _rows_in(metric: dict, op_id: int, nodes: list, measured: dict) -> int:
    """A stage's input rows, read off the stage below it when it could not count them.

    The streaming path meters a stage by wrapping its *output* generator, which sees no
    input — so it reports `rows_in=0` and the tree supplies it instead. In a linear chain
    (which is the only shape that path takes) a stage's input is exactly the output of the
    node directly beneath it, i.e. the next entry in the pre-order walk. Without this the
    table shows `0` for every streamed stage, which reads as "this stage consumed nothing"
    rather than "this seam could not observe it".
    """
    rows_in = int(metric.get("rows_in", 0))
    if rows_in:
        return rows_in
    child_id = op_id + 1
    if child_id >= len(nodes):
        return 0
    depth, _node = nodes[op_id]
    child_depth, _child = nodes[child_id]
    if child_depth != depth + 1:  # not this node's child — a sibling or an ancestor's
        return 0
    return int(measured.get(child_id, {}).get("rows_out", 0))
