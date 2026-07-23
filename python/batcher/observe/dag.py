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

from batcher.plan.profile import walk_ir

__all__ = ["build_dag"]

# Keys whose values are child plans, in the order they should appear left-to-right. Any
# other plan-valued key still becomes an edge (via the walk), but these are the ones whose
# ordering carries meaning — a join's build side must not swap sides on screen.
_ORDERED_CHILD_KEYS = ("left", "right", "input", "inputs")


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
        return {"nodes": [], "edges": [], "width": 0, "depth": 0}
    by_id = {int(op.get("op_id", -1)): op for op in ops}
    walked = list(walk_ir(ir))
    # Identity, not equality: two structurally identical scans are distinct nodes, and
    # dict equality would silently merge them into one.
    index = {id(node): op_id for op_id, (_depth, node) in enumerate(walked)}

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, int]] = []
    for op_id, (depth, node) in enumerate(walked):
        nodes.append(_node(op_id, depth, node, by_id.get(op_id)))
        for child in _children(node):
            child_id = index.get(id(child))
            if child_id is not None:
                edges.append({"from": child_id, "to": op_id})

    max_depth = max((n["depth"] for n in nodes), default=0)
    _assign_positions(nodes, edges, max_depth)
    width = max((n["column"] for n in nodes), default=0) + 1
    critical = _critical_path(nodes, edges)
    for node in nodes:
        node["on_critical_path"] = node["op_id"] in critical
    return {
        "nodes": nodes,
        "edges": edges,
        "width": width,
        "depth": max_depth + 1,
        "critical_path": critical,
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
    children: dict[int, list[int]] = {}
    for edge in edges:
        children.setdefault(edge["to"], []).append(edge["from"])
    cost = {n["op_id"]: float(n.get("elapsed_ms") or 0.0) for n in nodes}
    best_from: dict[int, tuple[float, list[int]]] = {}

    def walk(op_id: int) -> tuple[float, list[int]]:
        cached = best_from.get(op_id)
        if cached is not None:
            return cached
        kids = children.get(op_id, [])
        if not kids:
            result = (cost[op_id], [op_id])
        else:
            total, chain = max((walk(kid) for kid in kids), key=lambda pair: pair[0])
            result = (cost[op_id] + total, [op_id, *chain])
        best_from[op_id] = result
        return result

    # The root is the only node that is never a child.
    all_children = {edge["from"] for edge in edges}
    roots = [n["op_id"] for n in nodes if n["op_id"] not in all_children]
    if not roots:
        return []
    return max((walk(root) for root in roots), key=lambda pair: pair[0])[1]


def _children(node: Any) -> list[Any]:
    """The node's child plans, in draw order (named keys first, then any others)."""
    if not isinstance(node, dict):
        return []
    out: list[Any] = []
    seen: set[int] = set()
    for key in (*_ORDERED_CHILD_KEYS, *node.keys()):
        value = node.get(key)
        for candidate in value if isinstance(value, (list, tuple)) else [value]:
            if _is_plan(candidate) and id(candidate) not in seen:
                seen.add(id(candidate))
                out.append(candidate)
    return out


def _is_plan(value: Any) -> bool:
    """Whether an IR value is a relational node rather than an expression.

    A relational node has ``"op"``; an expression has ``"e"`` — and a *binary* expression
    has both (``{"e": "binary", "op": "gt"}``). So the test is ``"op"`` present and ``"e"``
    absent, matching `plan.profile`'s rule exactly; disagreeing would put a predicate on the
    canvas as though it were an operator.
    """
    return isinstance(value, dict) and "op" in value and "e" not in value


def _node(
    op_id: int, depth: int, node: dict[str, Any], op: dict[str, Any] | None
) -> dict[str, Any]:
    """One graph node: its identity, its plan detail, and its measurement if it ran."""
    kind = str(node.get("op", "?"))
    out: dict[str, Any] = {
        "op_id": op_id,
        "depth": depth,
        "kind": kind,
        "detail": _detail(kind, node),
        "measured": False,
        "rows_out": 0,
        "elapsed_ms": 0.0,
        "est_rows": None,
        "spilled": False,
        "spill_bytes": 0,
        "backend": "",
        "column": 0,
        "row": depth,
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


def _detail(kind: str, node: dict[str, Any]) -> str:
    """A short subtitle naming what *this* operator does — keys, columns, or predicate.

    The single most useful thing on a plan node after its type, and the reason the graph
    beats the text tree at a glance: "hash_join" alone never told anyone which join it was.
    """
    if kind == "hash_join":
        keys = ", ".join(str(k) for k in node.get("left_keys", []))
        return (
            f"{node.get('join_type', 'inner')} on {keys}"
            if keys
            else str(node.get("join_type", "inner"))
        )
    if kind == "aggregate":
        groups = [_alias(k) for k in node.get("group_keys", [])]
        aggs = [str(a.get("func", "?")) for a in node.get("aggregates", [])]
        by = f"by {', '.join(groups)}" if groups else "global"
        return f"{by} · {', '.join(aggs)}" if aggs else by
    if kind == "sort":
        keys = [_alias(k) for k in node.get("keys", [])]
        limit = node.get("limit")
        text = ", ".join(keys)
        return f"top {limit} by {text}" if limit else text
    if kind == "scan":
        return f"source {node.get('source_id', 0)}"
    if kind == "limit":
        return f"n = {node.get('n', node.get('limit', ''))}"
    if kind == "project":
        items = node.get("items") or node.get("projections") or []
        return f"{len(items)} columns" if items else ""
    if kind == "filter":
        return _expr_text(node.get("predicate"))
    return ""


def _alias(key: Any) -> str:
    """A sort/group key's display name, from its alias or its column expression."""
    if not isinstance(key, dict):
        return str(key)
    if key.get("alias"):
        return str(key["alias"])
    return _expr_text(key.get("expr")) or "?"


def _expr_text(expr: Any, depth: int = 0) -> str:
    """A compact one-line rendering of a scalar expression, truncated when deep.

    Deliberately shallow: this is a node subtitle, not an expression pretty-printer, and a
    deeply nested predicate rendered in full would blow out the node box.
    """
    if not isinstance(expr, dict) or depth > 2:
        return "…" if isinstance(expr, dict) else ""
    kind = expr.get("e")
    if kind == "col":
        return str(expr.get("name", "?"))
    if kind == "lit":
        value = expr.get("value")
        if isinstance(value, dict):
            return str(next(iter(value.values()), "?"))
        return str(value)
    if kind == "binary":
        left = _expr_text(expr.get("left"), depth + 1)
        right = _expr_text(expr.get("right"), depth + 1)
        return f"{left} {_OPERATORS.get(str(expr.get('op')), str(expr.get('op')))} {right}"
    return str(kind or "")


_OPERATORS = {
    "gt": ">",
    "lt": "<",
    "ge": "≥",
    "le": "≤",
    "eq": "=",
    "ne": "≠",
    "and": "AND",
    "or": "OR",
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
}


def _assign_positions(
    nodes: list[dict[str, Any]], edges: list[dict[str, int]], max_depth: int
) -> None:
    """Place each node on a (column, row) grid: sources at the bottom, root at the top.

    Leaves are dealt consecutive columns left-to-right in walk order, then every parent is
    centered over its children from the deepest layer up. That is the standard layered
    drawing for a tree, and a relational plan is a tree in every shape the engine builds.
    """
    children: dict[int, list[int]] = {}
    for edge in edges:
        children.setdefault(edge["to"], []).append(edge["from"])

    # Row 0 is the bottom of the drawing (the sources), so invert depth.
    for node in nodes:
        node["row"] = max_depth - node["depth"]

    next_column = 0
    for node in nodes:
        if not children.get(node["op_id"]):
            node["column"] = next_column
            next_column += 1

    by_id = {node["op_id"]: node for node in nodes}
    for node in sorted(nodes, key=lambda n: -n["depth"]):
        kids = children.get(node["op_id"])
        if kids:
            node["column"] = sum(by_id[k]["column"] for k in kids) / len(kids)
