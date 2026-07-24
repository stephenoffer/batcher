"""Reading a plan IR node: what is a child, and what does this operator actually do.

One implementation of "walk the children" and "name the operator", shared by the graph
builder, the EXPLAIN text renderer, and the plan differ. Three copies of `_expr_text` would
drift within a release, and the drift would show as a graph node and a text tree describing
the same operator differently — the exact failure that makes a reader stop trusting both.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ORDERED_CHILD_KEYS",
    "children",
    "describe",
    "expr_text",
    "is_plan",
    "kind_of",
]

# Keys whose values are child plans, in the order they should appear left-to-right. Any
# other plan-valued key still becomes an edge (via the walk), but these are the ones whose
# ordering carries meaning — a join's build side must not swap sides on screen.
ORDERED_CHILD_KEYS = ("left", "right", "input", "inputs")

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


def is_plan(value: Any) -> bool:
    """Whether an IR value is a relational node rather than an expression.

    A relational node has ``"op"``; an expression has ``"e"`` — and a *binary* expression
    has both (``{"e": "binary", "op": "gt"}``). So the test is ``"op"`` present and ``"e"``
    absent, matching `plan.profile`'s rule exactly; disagreeing would put a predicate on the
    canvas as though it were an operator.

    Args:
        value: Any value found inside an IR document.

    Returns:
        True when the value is a relational operator node.
    """
    return isinstance(value, dict) and "op" in value and "e" not in value


def kind_of(node: Any) -> str:
    """The operator tag of a plan node, or ``"?"`` when it has none.

    Args:
        node: A relational IR node.

    Returns:
        The ``op`` tag as a string.
    """
    return str(node.get("op", "?")) if isinstance(node, dict) else "?"


def children(node: Any) -> list[Any]:
    """The node's child plans, in draw order (named keys first, then any others).

    Args:
        node: A relational IR node.

    Returns:
        The child plan nodes, deduplicated by identity.
    """
    if not isinstance(node, dict):
        return []
    out: list[Any] = []
    seen: set[int] = set()
    for key in (*ORDERED_CHILD_KEYS, *node.keys()):
        value = node.get(key)
        for candidate in value if isinstance(value, (list, tuple)) else [value]:
            if is_plan(candidate) and id(candidate) not in seen:
                seen.add(id(candidate))
                out.append(candidate)
    return out


def describe(kind: str, node: dict[str, Any]) -> str:
    """A short subtitle naming what *this* operator does — keys, columns, or predicate.

    The single most useful thing on a plan node after its type, and the reason the graph
    beats the text tree at a glance: "hash_join" alone never told anyone which join it was.

    Args:
        kind: The operator's ``op`` tag.
        node: The operator's IR node.

    Returns:
        A one-line subtitle, or ``""`` when the operator has nothing worth naming.
    """
    handler = _DESCRIBERS.get(kind)
    return handler(node) if handler else ""


def _describe_join(node: dict[str, Any]) -> str:
    keys = ", ".join(str(k) for k in node.get("left_keys", []))
    join_type = str(node.get("join_type", "inner"))
    return f"{join_type} on {keys}" if keys else join_type


def _describe_aggregate(node: dict[str, Any]) -> str:
    groups = [_alias(k) for k in node.get("group_keys", [])]
    aggs = [str(a.get("func", "?")) for a in node.get("aggregates", [])]
    by = f"by {', '.join(groups)}" if groups else "global"
    return f"{by} · {', '.join(aggs)}" if aggs else by


def _describe_sort(node: dict[str, Any]) -> str:
    keys = [_alias(k) for k in node.get("keys", [])]
    limit = node.get("limit")
    text = ", ".join(keys)
    return f"top {limit} by {text}" if limit else text


def _describe_project(node: dict[str, Any]) -> str:
    items = node.get("items") or node.get("projections") or []
    return f"{len(items)} columns" if items else ""


#: One describer per operator tag. A dict rather than an if-chain so adding an operator is
#: one entry, and so the set of operators that *have* a subtitle is readable at a glance.
_DESCRIBERS = {
    "hash_join": _describe_join,
    "sort_merge_join": _describe_join,
    "asof_join": _describe_join,
    "aggregate": _describe_aggregate,
    "sort": _describe_sort,
    "scan": lambda node: f"source {node.get('source_id', 0)}",
    "limit": lambda node: f"n = {node.get('n', node.get('limit', ''))}",
    "project": _describe_project,
    "filter": lambda node: expr_text(node.get("predicate")),
    "window": _describe_aggregate,
    "distinct": lambda node: ", ".join(_alias(k) for k in node.get("keys", [])),
    "union": lambda node: "all" if node.get("all") else "distinct",
}


def _alias(key: Any) -> str:
    """A sort/group key's display name, from its alias or its column expression."""
    if not isinstance(key, dict):
        return str(key)
    if key.get("alias"):
        return str(key["alias"])
    return expr_text(key.get("expr")) or "?"


def expr_text(expr: Any, depth: int = 0) -> str:
    """A compact one-line rendering of a scalar expression, truncated when deep.

    Deliberately shallow: this is a node subtitle, not an expression pretty-printer, and a
    deeply nested predicate rendered in full would blow out the node box.

    Args:
        expr: A scalar expression IR node.
        depth: Current recursion depth, used to truncate.

    Returns:
        The rendered expression, ``"…"`` when truncated, or ``""`` for a non-expression.
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
        left = expr_text(expr.get("left"), depth + 1)
        right = expr_text(expr.get("right"), depth + 1)
        return f"{left} {_OPERATORS.get(str(expr.get('op')), str(expr.get('op')))} {right}"
    if kind == "not":
        return f"NOT {expr_text(expr.get('expr'), depth + 1)}"
    if kind == "call":
        args = ", ".join(expr_text(a, depth + 1) for a in expr.get("args", []))
        return f"{expr.get('name', '?')}({args})"
    return str(kind or "")
