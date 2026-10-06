"""Say whether a plan would translate to the device tier, and name what blocks it if not.

`explain(backend="gpu")` reads this. It answers on the driver, with no device: the structural
walk mirrors the matcher in `core.gpu_plan.tree` (every operator, join type and node kind must be
in the translated subset) but returns *which* node failed instead of a bare `None`, and a plan
that passes the walk is then rehearsed on zero-row frames by `translate._rehearsal_reason`, the
same check the GPU route runs before it starts a worker, which names an untranslated expression.

It is read-only against the translator: it calls the tier's own classifiers (`supported_op`,
`supported_aggregate`, `supported_window`, `DECLINED_OPS`, `JOIN_HOW`) and restates none of them,
so it cannot drift from what the route actually accepts. "Eligible" means the shape translates;
whether a query *runs* on a device also depends on `backend=`, a visible GPU, and Kyber's cost
policy, which this does not decide.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from batcher.io.source import Source
    from batcher.plan.logical import LogicalPlan

__all__ = ["annotate_explain", "device_verdict"]


def annotate_explain(rendered: str, plan: LogicalPlan, sources: list[Source], fmt: str) -> str:
    """Add the device-tier verdict to an `explain()` rendering.

    Args:
        rendered: The plan as `explain` rendered it, text or a JSON document.
        plan: The logical plan as the user built it.
        sources: The query's sources.
        fmt: ``"text"`` to append one line, ``"json"`` to add a ``"device"`` object.

    Returns:
        The rendering with the verdict added.
    """
    reason = device_verdict(plan, sources)
    if fmt == "json":
        import json

        doc = json.loads(rendered)
        doc["device"] = {"eligible": reason is None, "reason": reason}
        return json.dumps(doc, indent=2, default=str)
    verdict = "eligible" if reason is None else f"declined: {reason}"
    return f"{rendered.rstrip()}\ndevice tier (backend='gpu'): {verdict}\n"


def device_verdict(plan: LogicalPlan, sources: list[Source]) -> str | None:
    """Why `plan` would not translate to the device tier, or `None` when it would.

    The plan is optimized first, exactly as the GPU route does before translating, so the
    verdict is about the shape the device would actually be handed.

    Args:
        plan: The logical plan as the user built it.
        sources: The query's sources, indexed by a scan's `source_id`.

    Returns:
        A reason naming the blocking operator or expression, or `None` when the plan's shape
        translates and its zero-row rehearsal raised no decline.
    """
    from batcher import core
    from batcher.api.terminal.gpu_backend.route import _optimized
    from batcher.api.terminal.gpu_backend.translate import _rehearsal_reason

    plan = _optimized(plan, sources, core.default_hub())
    reason = _structural_reason(plan)
    if reason is not None:
        return reason
    rehearsed = _rehearsal_reason(plan, sources)
    if rehearsed is None:
        return None
    from batcher.core.gpu_plan.exprs import DECLINED_EXPRS

    # `eval_expr` declines a tag as "expr <tag>"; the tier records why for every such tag.
    tag = rehearsed.removeprefix("expr ")
    why = f" ({DECLINED_EXPRS[tag]})" if tag != rehearsed and tag in DECLINED_EXPRS else ""
    return f"an expression is not translated: {rehearsed}{why}"


def _structural_reason(plan: LogicalPlan) -> str | None:
    """The first node, top-down, that keeps the plan out of the translated tree shape."""
    from batcher.core.gpu_plan.eligibility import JOIN_HOW
    from batcher.core.gpu_plan.ops import DECLINED_OPS, supported_op
    from batcher.plan.logical import Join, Scan, Union

    node: Any = plan
    while not isinstance(node, Scan | Join | Union):
        if node is None:
            return "the plan reads from a leaf that is not a source scan"
        try:
            ir = node.to_ir()
        except NotImplementedError:
            return (
                f"{type(node).__name__} is a Python-only stage (a callable such as map_batches), "
                "which has no engine IR to translate"
            )
        if not supported_op(ir):
            return _operator_reason(ir, DECLINED_OPS)
        node = getattr(node, "input", None)
    if isinstance(node, Scan):
        return None
    if isinstance(node, Union):
        return next((r for r in map(_structural_reason, node.inputs) if r is not None), None)
    reason = _structural_reason(node.left) or _structural_reason(node.right)
    if reason is not None:
        return reason
    join_ir = node.to_ir()
    op = join_ir.get("op")
    if op != "hash_join":
        return f"operator {op!r}: {DECLINED_OPS.get(op, 'not translated')}"
    join_type = join_ir.get("join_type")
    if join_type not in JOIN_HOW:
        return f"join type {join_type!r} is not translated (supported: {sorted(JOIN_HOW)})"
    return None


def _operator_reason(ir: dict, declined: dict[str, str]) -> str:
    """Name the part of one untranslatable operator node that is outside the subset."""
    from batcher.core.gpu_plan.aggs import supported_aggregate
    from batcher.core.gpu_plan.windows import supported_window

    op = ir.get("op")
    if op in declined:
        return f"operator {op!r}: {declined[op]}"
    if op == "aggregate":
        bad = [a for a in ir["aggregates"] if not supported_aggregate({**ir, "aggregates": [a]})]
        named = ", ".join(_agg_label(a) for a in bad)
        return f"operator 'aggregate': the reduction(s) {named} are not translated"
    if op == "window":
        bad = [f for f in ir["functions"] if not supported_window({**ir, "functions": [f]})]
        if bad:
            named = ", ".join(repr(f.get("func")) for f in bad)
            return f"operator 'window': the function(s) {named} are not translated in this frame"
        return (
            "operator 'window': its shape is not translated (a per-partition top-N, or order "
            "keys that place nulls differently)"
        )
    if op == "distinct":
        return "operator 'distinct': a keyed distinct (subset/keep) is not translated"
    return f"operator {op!r} is not translated"


def _agg_label(agg: dict) -> str:
    """One aggregate's function name, with the option that declined it when it is not the name."""
    func = repr(agg.get("func"))
    if agg.get("interpolation") is not None:
        return f"{func} (interpolation={agg['interpolation']!r})"
    if agg.get("order_by"):
        return f"{func} (order_by)"
    return func
