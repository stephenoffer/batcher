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


def expr_text(expr: Any, depth: int = 0, max_depth: int = 2) -> str:
    """A compact one-line rendering of a scalar expression, truncated when deep.

    Deliberately shallow by default: this is a node subtitle, not an expression
    pretty-printer, and a deeply nested predicate rendered in full would blow out the node
    box. A caller rendering into a line rather than a box raises `max_depth`.

    Two levels is not enough for a *pushed* predicate, which is why the default is not the
    only option. The optimizer routinely brackets a set membership with the bounds it
    derives for zone-map pruning, turning ``c IN (…)`` into a three-deep conjunction — and
    truncating that at depth two elides the column names, leaving ``… ≤ … AND … IN (…)``,
    which names nothing the reader was asking about.

    Args:
        expr: A scalar expression IR node.
        depth: Current recursion depth, used to truncate.
        max_depth: Deepest level rendered before collapsing to ``"…"``.

    Returns:
        The rendered expression, ``"…"`` when truncated, or ``""`` for a non-expression.
    """
    if not isinstance(expr, dict) or depth > max_depth:
        return "…" if isinstance(expr, dict) else ""
    kind = expr.get("e")
    if kind == "col":
        return str(expr.get("name", "?"))
    if kind == "lit":
        return _lit_text(expr.get("value"))
    if kind == "binary":
        left = expr_text(expr.get("left"), depth + 1, max_depth)
        right = expr_text(expr.get("right"), depth + 1, max_depth)
        return f"{left} {_OPERATORS.get(str(expr.get('op')), str(expr.get('op')))} {right}"
    if kind == "not":
        # The operand is `input`, as it is on every other unary node. Reading `expr` here
        # rendered every negated predicate as a bare ``NOT`` with nothing after it.
        return f"NOT ({expr_text(expr.get('input'), depth + 1, max_depth)})"
    if kind in ("is_null", "is_not_null"):
        null_test = "IS NULL" if kind == "is_null" else "IS NOT NULL"
        return f"{expr_text(expr.get('input'), depth + 1, max_depth)} {null_test}"
    if kind == "in_list":
        return _in_list_text(expr, depth, max_depth)
    if kind == "str":
        pattern = expr.get("pattern")
        target = expr_text(expr.get("input"), depth + 1, max_depth)
        shown = f"{target}, {pattern!r}" if isinstance(pattern, str) else target
        return f"{expr.get('fn', 'str')}({shown})"
    if kind == "cast":
        return f"{expr_text(expr.get('input'), depth + 1, max_depth)}::{expr.get('dtype', '?')}"
    if kind == "call":
        args = ", ".join(expr_text(a, depth + 1, max_depth) for a in expr.get("args", []))
        return f"{expr.get('name', '?')}({args})"
    return str(kind or "")


#: Members of an `IN` list rendered before the rest are summarized as a count. A pushed set
#: is routinely hundreds of keys long, and a node subtitle that prints all of them stops
#: being a subtitle.
_IN_LIST_SHOWN = 4


def _lit_text(value: Any) -> str:
    """A literal's display text, from either the tagged IR form or a bare Python value."""
    if isinstance(value, dict):
        return str(next(iter(value.values()), "?"))
    return str(value)


def _in_list_text(expr: dict[str, Any], depth: int, max_depth: int) -> str:
    """``x IN (a, b, … 12 more)`` for a set-membership node."""
    members = expr.get("set") or []
    shown = ", ".join(_lit_text(m) for m in members[:_IN_LIST_SHOWN])
    if len(members) > _IN_LIST_SHOWN:
        shown += f", … {len(members) - _IN_LIST_SHOWN} more"
    return f"{expr_text(expr.get('input'), depth + 1, max_depth)} IN ({shown})"
