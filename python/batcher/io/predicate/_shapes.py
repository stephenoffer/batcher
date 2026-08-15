"""Recognizers for the predicate shapes the translators share.

Each returns the parts of a node a backend needs, or None when the node is not that
shape, so the five translators agree on what an `IN` list or a string predicate *is*
without restating the structural checks. `_combine` is here for the same reason: the
widening rule for a partly-translated `AND`/`OR` is one decision, not five.
"""

from __future__ import annotations

import math
from typing import Any

#: String predicates expressible as a backend pattern match.
_STR_PUSHABLE = frozenset({"starts_with", "ends_with", "contains"})


def _const_bool(ir: dict[str, Any]) -> bool | None:
    """The value of a constant boolean predicate, or None if `ir` is not one.

    Worth translating because the expression builder folds degenerate predicates to a
    constant rather than leaving them for the engine: ``col.is_in([])`` lowers to a bare
    ``lit False``. Pushed, that prunes the scan to nothing instead of reading the relation
    to discard every row.
    """
    if ir.get("e") != "lit":
        return None
    value = ir.get("value")
    if not isinstance(value, dict) or "bool" not in value:
        return None
    return bool(value["bool"])


def _in_list(ir: dict[str, Any]) -> tuple[str, list[dict[str, Any]]] | None:
    """Return ``(column, [literal_ir, ...])`` for a pushable `IN` list, else None.

    The set never holds a null and is never empty by the time it reaches here — the
    builder folds ``is_in([])`` to a constant `False` and splits a null member out into an
    ``OR`` with a `nullif` term, because SQL's ``x IN (NULL)`` is `NULL` rather than true.
    Both are re-checked anyway so this stays correct against an IR built by hand or by the
    SQL parser.
    """
    if ir.get("e") != "in_list" or ir.get("input", {}).get("e") != "col":
        return None
    members = ir.get("set")
    if not members:
        return None
    for member in members:
        if not isinstance(member, dict) or len(member) != 1:
            return None
        ((kind, value),) = member.items()
        if kind == "null" or value is None:
            return None
        if isinstance(value, float) and not math.isfinite(value):
            return None
    return ir["input"]["name"], list(members)


def _str_predicate(ir: dict[str, Any]) -> tuple[str, str, str] | None:
    """Return ``(column, fn, pattern)`` for a pushable string predicate, else None."""
    if ir.get("e") != "str" or ir.get("fn") not in _STR_PUSHABLE:
        return None
    pattern = ir.get("pattern")
    if ir.get("input", {}).get("e") != "col" or not isinstance(pattern, str):
        return None
    return ir["input"]["name"], ir["fn"], pattern


def _combine(op: str, left: Any, right: Any, both: Any) -> Any | None:
    """Fold a translated `AND`/`OR` pair, keeping a partial conjunction.

    `both` is the already-combined value, evaluated by the caller only when neither side
    is None. An `AND` degrades to whichever side translated; an `OR` declines unless both
    did. See this module's docstring for why the two directions differ.

    Args:
        op: ``"and"`` or ``"or"``.
        left: The translated left operand, or None.
        right: The translated right operand, or None.
        both: The combination of the two, used when both translated.

    Returns:
        The folded filter, or None when nothing can be pushed.
    """
    if left is not None and right is not None:
        return both
    if op == "or":
        return None
    return left if left is not None else right
