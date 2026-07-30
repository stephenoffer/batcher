"""Which plans the GPU translator can run — the matcher in front of the kernels.

A plan is GPU-eligible only when *every* node in it translates, so this decides how much of a
real workload reaches the device. Three shapes are recognized, each returning the pieces the
executor replays and `None` otherwise, which the caller reads as "use the CPU engine":

* a linear chain of operators over one scan;
* an equi-join of two chains, each over its own scan, plus a chain above the join;
* a union of scans plus a chain above it.

The join and union forms accept a **chain on each input** rather than a bare scan. That is not
a cosmetic generalization: the optimizer pushes filters and projections down to just above the
scans, so the pushed-down form of essentially every real join has a `filter` or `project`
between the join and its inputs — a matcher requiring bare scans matches the *unoptimized*
shape and almost never the one it is actually handed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher.core.gpu_plan.ops import supported_op

if TYPE_CHECKING:
    from batcher.plan.logical import LogicalPlan

__all__ = ["JOIN_HOW", "gpu_join_spec", "gpu_plan_ops", "gpu_union_spec"]

#: Join types the translator runs, mapped to the backends' `merge` mode. `semi` and `anti`
#: have no `merge` mode and are handled as key-membership filters by the executor.
JOIN_HOW = {
    "inner": "inner",
    "left": "left",
    "right": "right",
    "outer": "outer",
    "full": "outer",
    "semi": "semi",
    "anti": "anti",
}


def _chain(node: Any, stop: tuple) -> tuple[Any, list[dict]] | None:
    """Walk down from `node` collecting translatable operators until a `stop` node.

    Returns `(stop_node, [op_ir, ...])` bottom-up (nearest the leaf first), or `None` when
    anything on the way down is untranslatable. A node whose `to_ir()` raises is Python-only
    (a `map_batches` UDF), which is exactly a "not lowered to the engine IR" answer.
    """
    ops: list[dict] = []
    while node is not None and not isinstance(node, stop):
        try:
            ir = node.to_ir()
        except Exception:
            return None
        if not supported_op(ir):
            return None
        ops.append(ir)
        node = getattr(node, "input", None)
    if node is None:
        return None
    ops.reverse()
    return node, ops


def gpu_plan_ops(plan: LogicalPlan):
    """`(scan, [op_ir, ...])` for a linear chain of translatable operators over one scan.

    Args:
        plan: The logical plan to match.

    Returns:
        The scan and its bottom-up operator chain, or `None` when the plan is not a
        translatable single-source chain (the caller then uses the CPU engine).
    """
    from batcher.plan.logical import Scan

    matched = _chain(plan, (Scan,))
    if matched is None:
        return None
    scan, ops = matched
    return (scan, ops) if ops else None


def gpu_join_spec(plan: LogicalPlan):
    """`(left, right, join_ir, ops)` for `[ops] over Join(chain, chain)`, else `None`.

    `left` and `right` are each `(scan, [op_ir, ...])`, so a filter or projection pushed
    below the join runs on the device with it rather than forcing the whole plan to the host.

    Args:
        plan: The logical plan to match.

    Returns:
        The two input chains, the join's IR, and the operator chain above it, or `None`.
    """
    from batcher.plan.logical import Join, Scan

    matched = _chain(plan, (Scan, Join))
    if matched is None:
        return None
    node, ops = matched
    if not isinstance(node, Join):
        return None
    left = _chain(node.left, (Scan,))
    right = _chain(node.right, (Scan,))
    if left is None or right is None:
        return None
    join_ir = node.to_ir()
    if join_ir.get("op") != "hash_join" or join_ir.get("join_type") not in JOIN_HOW:
        return None
    return left, right, join_ir, ops


def gpu_union_spec(plan: LogicalPlan):
    """`(inputs, distinct, ops)` for `[ops] over Union(chains)`, else `None`.

    Args:
        plan: The logical plan to match.

    Returns:
        Each input's `(scan, [op_ir, ...])` chain, whether the union deduplicates, and the
        operator chain above it, or `None`.
    """
    from batcher.plan.logical import Scan, Union

    matched = _chain(plan, (Scan, Union))
    if matched is None:
        return None
    node, ops = matched
    if not isinstance(node, Union):
        return None
    inputs = [_chain(i, (Scan,)) for i in node.inputs]
    if any(i is None for i in inputs):
        return None
    return inputs, bool(node.to_ir().get("distinct", False)), ops
