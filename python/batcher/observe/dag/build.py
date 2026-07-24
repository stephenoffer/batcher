"""The executed plan as a laid-out graph — nodes, edges, and their measured stats.

Turns the optimized IR (a nested `to_ir()` document) into the flat `{nodes, edges}` shape
the dashboard draws, with each node carrying the operator's measured rows, time, and spill
verdict. The layout is computed here rather than in the browser so the server owns one
answer to "what shape is this plan", and the page stays a renderer.

**`op_id` comes from `plan.profile.walk_ir`, never a local walk.** The pre-order index of
that walk *is* the operator's identity — the engine's metrics, Kyber's estimates, and this
graph all key off it. A second walk that ordered children differently would draw a plan
whose nodes were labeled with another operator's numbers, which is worse than drawing
nothing: it is confidently wrong.

Layout is bottom-up by depth: sources at the bottom, the root at the top, matching how a
plan is read and how Spark and DuckDB draw theirs. Siblings are spread within their layer
and parents centered over their children, which is enough for the left-deep and bushy join
trees a relational plan actually produces — this is a plan, not an arbitrary graph, so a
force simulation would be motion without information.
"""

from __future__ import annotations

from typing import Any

from batcher.observe.dag.describe import children, describe, kind_of
from batcher.plan.profile import walk_ir

__all__ = ["BREAKERS", "build_dag", "plan_shape"]

#: The node fields a thumbnail needs — its identity, what it is, and where it sits. Nothing
#: measured: a pipeline thumbnail is a *fingerprint of the shape*, drawn from the plan alone,
#: so it renders identically whether or not any run has profiled it yet.
_SHAPE_FIELDS = ("op_id", "kind", "detail", "column", "row", "breaker", "on_critical_path")


def plan_shape(ir: dict[str, Any] | None) -> dict[str, Any]:
    """A pipeline's plan as a compact, unmeasured graph, for a thumbnail-sized drawing.

    The same layout as `build_dag` with the per-operator measurements stripped out, so the
    pipelines list can carry one small graph per pipeline — the visual fingerprint that lets
    a reader tell one pipeline from another at a glance without opening it.

    Args:
        ir: The plan IR, or None.

    Returns:
        ``{"nodes": [...], "edges": [...], "width": int, "depth": int}`` with only the
        layout and identity fields on each node; empty lists when `ir` is None.
    """
    full = build_dag(ir, [])
    return {
        "nodes": [{k: node[k] for k in _SHAPE_FIELDS} for node in full["nodes"]],
        "edges": full["edges"],
        "width": full["width"],
        "depth": full["depth"],
    }


#: Operators that must materialize their whole input before emitting a row. They are where
#: a pipeline breaks into stages, which is what the timeline and the adaptive layer key off:
#: everything between two breakers streams concurrently, so it belongs to one stage.
BREAKERS = frozenset({"aggregate", "sort", "hash_join", "sort_merge_join", "distinct", "window"})


def build_dag(ir: dict[str, Any] | None, ops: list[dict[str, Any]]) -> dict[str, Any]:
    """Lay out `ir` as a graph whose nodes carry the measured stats from `ops`.

    Args:
        ir: The optimized plan IR, or None when the query never reached the optimizer.
        ops: The per-operator profile dicts, keyed by ``op_id``.

    Returns:
        ``{"nodes": [...], "edges": [...], "width": int, "depth": int}``. Empty lists when
        `ir` is None, so the caller renders "no plan" rather than special-casing.
    """
    if not ir:
        return {
            "nodes": [],
            "edges": [],
            "width": 0,
            "depth": 0,
            "critical_path": [],
            "stages": 0,
        }
    by_id = {int(op.get("op_id", -1)): op for op in ops}
    walked = list(walk_ir(ir))
    # Identity, not equality: two structurally identical scans are distinct nodes, and
    # dict equality would silently merge them into one.
    index = {id(node): op_id for op_id, (_depth, node) in enumerate(walked)}

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, int]] = []
    for op_id, (depth, node) in enumerate(walked):
        nodes.append(_node(op_id, depth, node, by_id.get(op_id)))
        for child in children(node):
            child_id = index.get(id(child))
            if child_id is not None:
                edges.append({"from": child_id, "to": op_id})

    max_depth = max((n["depth"] for n in nodes), default=0)
    _assign_positions(nodes, edges, max_depth)
    width = max((n["column"] for n in nodes), default=0) + 1
    critical = _critical_path(nodes, edges)
    for node in nodes:
        node["on_critical_path"] = node["op_id"] in critical
    _assign_stages(nodes, edges)
    return {
        "nodes": nodes,
        "edges": edges,
        "width": width,
        "depth": max_depth + 1,
        "critical_path": critical,
        "stages": max((int(n["stage"] or 0) for n in nodes), default=0) + 1,
    }


def _critical_path(nodes: list[dict[str, Any]], edges: list[dict[str, int]]) -> list[int]:
    """The root-to-leaf chain whose summed operator time is largest.

    The answer to "if I made one thing faster, which chain would it have to be on". Every
    other operator can overlap with something else, but this chain's steps feed each other,
    so its total is the floor on the plan's latency. Highlighting it turns a graph of twenty
    boxes into a shortlist of four.

    Computed by longest-path DP over the tree, from the root down: a plan is a DAG whose
    edges point child -> parent, so the recursion is well-founded and needs no cycle guard.
    """
    if not nodes:
        return []
    kids = _child_map(edges)
    cost = {n["op_id"]: float(n.get("elapsed_ms") or 0.0) for n in nodes}
    best_from: dict[int, tuple[float, list[int]]] = {}

    def walk(op_id: int) -> tuple[float, list[int]]:
        cached = best_from.get(op_id)
        if cached is not None:
            return cached
        below = kids.get(op_id, [])
        if not below:
            result = (cost[op_id], [op_id])
        else:
            total, chain = max((walk(kid) for kid in below), key=lambda pair: pair[0])
            result = (cost[op_id] + total, [op_id, *chain])
        best_from[op_id] = result
        return result

    # The root is the only node that is never a child.
    all_children = {edge["from"] for edge in edges}
    roots = [n["op_id"] for n in nodes if n["op_id"] not in all_children]
    if not roots:
        return []
    return max((walk(root) for root in roots), key=lambda pair: pair[0])[1]


def _child_map(edges: list[dict[str, int]]) -> dict[int, list[int]]:
    """Parent op_id -> the op_ids feeding it."""
    kids: dict[int, list[int]] = {}
    for edge in edges:
        kids.setdefault(edge["to"], []).append(edge["from"])
    return kids


def _assign_stages(nodes: list[dict[str, Any]], edges: list[dict[str, int]]) -> None:
    """Tag each node with the pipeline stage it belongs to, sources in stage 0.

    A relational plan is not a sequence of steps: operators between two pipeline breakers
    stream into each other and run at the same time. Numbering the stages is what lets a
    timeline lay concurrent work side by side instead of stacking it as though it were
    sequential, which is the commonest way an operator timeline lies. It is also the unit
    the adaptive layer re-optimizes at, so the same number names the same thing in both.
    """
    kids = _child_map(edges)
    by_id = {n["op_id"]: n for n in nodes}

    def stage_of(op_id: int) -> int:
        node = by_id[op_id]
        cached = node.get("stage")
        if cached is not None:
            return int(cached)
        # Provisional value before recursing: a malformed IR with a cycle would otherwise
        # recurse forever, and observability must never hang the process it observes.
        node["stage"] = 0
        below = kids.get(op_id, [])
        base = max((stage_of(k) for k in below), default=-1)
        # A breaker ends the stage beneath it and starts a new one; a streaming operator
        # joins the stage its input already belongs to.
        stage = base + 1 if (node["kind"] in BREAKERS or not below) else max(base, 0)
        node["stage"] = stage
        return stage

    for node in nodes:
        node["stage"] = None
    for node in nodes:
        stage_of(node["op_id"])


def _node(
    op_id: int, depth: int, node: dict[str, Any], op: dict[str, Any] | None
) -> dict[str, Any]:
    """One graph node: its identity, its plan detail, and its measurement if it ran."""
    kind = kind_of(node)
    out: dict[str, Any] = {
        "op_id": op_id,
        "depth": depth,
        "kind": kind,
        "detail": describe(kind, node),
        "measured": False,
        "rows_out": 0,
        "elapsed_ms": 0.0,
        "est_rows": None,
        "spilled": False,
        "spill_bytes": 0,
        "backend": "",
        "column": 0,
        "row": depth,
        "breaker": kind in BREAKERS,
        "stage": None,
    }
    if op:
        # Everything the engine measured for this operator. The dashboard's operator table
        # and node inspector are the only readers of most of these, and a field omitted here
        # is a field no amount of front-end work can show.
        out.update(
            measured=bool(op.get("measured")),
            rows_in=int(op.get("rows_in", 0)),
            rows_out=int(op.get("rows_out", 0)),
            elapsed_ms=float(op.get("elapsed_ms", 0.0)),
            est_rows=op.get("est_rows"),
            est_error=op.get("est_error"),
            selectivity=op.get("selectivity"),
            spilled=bool(op.get("spilled")),
            spill_bytes=int(op.get("spill_bytes", 0)),
            result_bytes=int(op.get("result_bytes", 0)),
            peak_rss_bytes=int(op.get("peak_rss_bytes", 0)),
            cpu_util=float(op.get("cpu_util", 0.0)),
            threads=int(op.get("threads", 0)),
            backend=str(op.get("backend", "")),
            algorithm=str(op.get("algorithm", "")),
            provenance=str(op.get("provenance", "")),
        )
    return out


def _assign_positions(
    nodes: list[dict[str, Any]], edges: list[dict[str, int]], max_depth: int
) -> None:
    """Place each node on a (column, row) grid: sources at the bottom, root at the top.

    Leaves are dealt consecutive columns left-to-right in walk order, then every parent is
    centered over its children from the deepest layer up. That is the standard layered
    drawing for a tree, and a relational plan is a tree in every shape the engine builds.
    """
    kids = _child_map(edges)

    # Row 0 is the bottom of the drawing (the sources), so invert depth.
    for node in nodes:
        node["row"] = max_depth - node["depth"]

    next_column = 0
    for node in nodes:
        if not kids.get(node["op_id"]):
            node["column"] = next_column
            next_column += 1

    by_id = {node["op_id"]: node for node in nodes}
    for node in sorted(nodes, key=lambda n: -n["depth"]):
        below = kids.get(node["op_id"])
        if below:
            node["column"] = sum(by_id[k]["column"] for k in below) / len(below)
