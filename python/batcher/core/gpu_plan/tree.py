"""The whole-plan form of the translator: any tree of scans, joins and unions on the device.

`eligibility` matches three *fixed* shapes — a chain, one join, one union — and returns `None`
for everything else. That ceiling is not a detail of the matcher; it is the thing that decided
how much of a real workload ever reached a device. Measured against the 22-query TPC-H suite,
nine queries matched and thirteen went to the CPU engine, and every one of the thirteen was
refused for the same reason: it joins three relations instead of two.

This states the shape recursively instead, so a plan is eligible when *every node in it* is,
which is the only rule that was ever meant. A leaf is a scan, an internal node is a join or a
union, and every node carries the run of linear operators sitting directly above it — the
filters and projections the optimizer pushes down, the aggregate a sub-plan reduces with. The
result is a plain JSON document, because it has to travel to a Ray worker intact.

The kernels are unchanged. `run_tree` composes `run_ops`, `join_frames` and `union_frames`,
which is deliberate: the linear matcher's join and this one's *are* the same join, and the way
two translators for one operator disagree is not a crash but a different answer on the null key
or the negative zero. Nothing here knows what a join is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher.core.gpu_plan.eligibility import JOIN_HOW
from batcher.core.gpu_plan.ops import supported_op

if TYPE_CHECKING:
    from batcher.core.gpu_plan.backend import DfBackend
    from batcher.plan.logical import LogicalPlan

__all__ = [
    "gpu_tree_spec",
    "run_tree",
    "tree_leaves",
    "tree_scan_ops",
    "tree_size",
]


def gpu_tree_spec(plan: LogicalPlan) -> tuple[dict, list] | None:
    """`(spec, scans)` for any translatable plan tree, or `None` when some node is not.

    The spec is JSON-serializable so it can be a Ray task argument. Each leaf carries a `leaf`
    index into `scans`, assigned in the order the leaves are encountered, so a self-join — two
    leaves over one source, which is what TPC-H q21 is — keeps its two inputs distinct even
    though `source_id` is the same for both.

    Args:
        plan: The logical plan to match.

    Returns:
        The tree spec and the `Scan` nodes its leaves refer to, or `None` when any operator,
        join type or node kind in the plan is outside the translated subset.
    """
    scans: list = []
    spec = _node(plan, scans)
    return None if spec is None else (spec, scans)


def _node(plan: LogicalPlan, scans: list) -> dict | None:
    """One tree node: the run of linear operators above `plan`, plus what they sit on."""
    from batcher.plan.logical import Join, Scan, Union

    ops: list[dict] = []
    node: Any = plan
    while node is not None and not isinstance(node, (Scan, Join, Union)):
        try:
            ir = node.to_ir()
        except Exception:
            # Not lowered to the engine IR at all — a `map_batches` UDF is Python-only, which
            # is exactly a "the device cannot run this" answer rather than an error.
            return None
        if not supported_op(ir):
            return None
        ops.append(ir)
        node = getattr(node, "input", None)
    if node is None:
        return None
    ops.reverse()
    if isinstance(node, Scan):
        scans.append(node)
        return {"kind": "scan", "leaf": len(scans) - 1, "source_id": node.source_id, "ops": ops}
    if isinstance(node, Join):
        return _join_node(node, ops, scans)
    return _union_node(node, ops, scans)


def _join_node(node, ops: list[dict], scans: list) -> dict | None:
    """A `join` tree node, or `None` when its type or either input is untranslatable.

    The two inputs are matched left before right so the leaf numbering follows the plan's own
    reading order, which is what makes a spec reproducible across runs — and reproducible is
    what lets a descriptor list built once be reused by a retry.
    """
    # A node's `to_ir` lowers its whole subtree, so a Python-only stage anywhere below this join
    # raises here rather than at the leaf that holds it. Declining is the same answer either way.
    try:
        join_ir = node.to_ir()
    except Exception:
        return None
    if join_ir.get("op") != "hash_join" or join_ir.get("join_type") not in JOIN_HOW:
        return None
    left = _node(node.left, scans)
    if left is None:
        return None
    right = _node(node.right, scans)
    if right is None:
        return None
    return {
        "kind": "join",
        "left": left,
        "right": right,
        "join": {k: v for k, v in join_ir.items() if k not in ("left", "right")},
        "ops": ops,
    }


def _union_node(node, ops: list[dict], scans: list) -> dict | None:
    """A `union` tree node, or `None` when any input is untranslatable."""
    try:
        distinct = bool(node.to_ir().get("distinct", False))
    except Exception:
        return None
    inputs: list[dict] = []
    for child in node.inputs:
        spec = _node(child, scans)
        if spec is None:
            return None
        inputs.append(spec)
    return {"kind": "union", "inputs": inputs, "distinct": distinct, "ops": ops}


def tree_leaves(spec: dict) -> list[dict]:
    """Every `scan` node in the tree, in leaf-index order.

    Args:
        spec: A tree spec from `gpu_tree_spec`.

    Returns:
        The leaf nodes, so a caller can size each one and decide which to shard.
    """
    out: list[dict] = []
    _collect_leaves(spec, out)
    out.sort(key=lambda leaf: leaf["leaf"])
    return out


def _collect_leaves(spec: dict, out: list[dict]) -> None:
    kind = spec["kind"]
    if kind == "scan":
        out.append(spec)
        return
    if kind == "join":
        _collect_leaves(spec["left"], out)
        _collect_leaves(spec["right"], out)
        return
    for child in spec["inputs"]:
        _collect_leaves(child, out)


def tree_size(spec: dict) -> int:
    """How many nodes the tree has — the cheap "is this worth a tree at all" question."""
    kind = spec["kind"]
    if kind == "scan":
        return 1
    if kind == "join":
        return 1 + tree_size(spec["left"]) + tree_size(spec["right"])
    return 1 + sum(tree_size(child) for child in spec["inputs"])


def tree_scan_ops(spec: dict) -> tuple[dict, list[dict]] | None:
    """`(leaf, ops)` when the whole tree is one leaf, else `None`.

    The linear matcher's shape, recognized on the tree so a caller can keep using the
    single-source fan-out — which reads its shard on the device and folds a mergeable
    reducer — rather than paying the tree path's broadcast machinery for a plan with
    nothing to broadcast.
    """
    return (spec, spec["ops"]) if spec["kind"] == "scan" else None


def run_tree(spec: dict, frames: dict, be: DfBackend):
    """Execute a tree spec against already-read leaf frames.

    Args:
        spec: A tree spec from `gpu_tree_spec`.
        frames: Leaf index -> the frame that leaf's scan produced, already on `be`.
        be: The dataframe backend to compute on.

    Returns:
        The tree's result, as a frame on `be`.

    Raises:
        Unsupported: For an expression outside the translated subset, which every caller turns
            into a CPU-engine fallback.
    """
    from batcher.core.gpu_plan.execute import join_frames, run_ops, union_frames

    kind = spec["kind"]
    if kind == "scan":
        df = frames[spec["leaf"]]
    elif kind == "join":
        # Depth-first, and the build side last: the probe side's frame is the one a filter or a
        # projection above the scan has already shrunk, so evaluating it first means the peak
        # holds one reduced input rather than two raw ones.
        left = run_tree(spec["left"], frames, be)
        right = run_tree(spec["right"], frames, be)
        df = join_frames(left, right, spec["join"], be)
    else:
        df = union_frames([run_tree(c, frames, be) for c in spec["inputs"]], spec["distinct"], be)
    return run_ops(df, spec["ops"], be)
