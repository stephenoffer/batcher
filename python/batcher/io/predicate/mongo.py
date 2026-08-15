"""IR to a MongoDB filter document, for the Mongo source.

Document stores match on a query document rather than an expression tree, so negation and
the string predicates take shapes the other backends do not have: `$nor` for a
whole-expression `NOT`, and an anchored, escaped `$regex` for a prefix.
"""

from __future__ import annotations

import re
from typing import Any

from batcher.io.predicate._literals import _col_and_literal, _literal
from batcher.io.predicate._shapes import _combine, _const_bool, _in_list, _str_predicate
from batcher.plan.ir_tags import COMPARISON_FLIP, COMPARISON_OPS

__all__ = ["to_mongo_filter"]

_MONGO_OP = {"eq": "$eq", "ne": "$ne", "lt": "$lt", "le": "$lte", "gt": "$gt", "ge": "$gte"}


def _mongo_str(ir: dict[str, Any]) -> dict[str, Any] | None:
    """A MongoDB `$regex` document for a pushable string predicate, or None.

    The pattern is `re.escape`-d, so a value containing regex metacharacters matches
    literally rather than as a pattern — the difference between finding the documents
    whose ``path`` starts with ``a.b`` and finding those whose ``path`` starts with ``a``
    followed by anything. Anchored for `starts_with`/`ends_with`, which is also what lets
    the server use an index for the prefix case instead of scanning.
    """
    parsed = _str_predicate(ir)
    if parsed is None:
        return None
    column, fn, pattern = parsed
    quoted = re.escape(pattern)
    expression = {
        "starts_with": f"^{quoted}",
        "ends_with": f"{quoted}$",
        "contains": quoted,
    }[fn]
    return {column: {"$regex": expression}}


def to_mongo_filter(ir: dict[str, Any], *, exact: bool = False) -> dict[str, Any] | None:
    """Translate the pushable subset of `ir` to a MongoDB filter document, or None.

    Args:
        ir: The predicate's IR dictionary.
        exact: Decline any term that would only widen the match, so the document selects
            exactly the predicate's rows. Set by the recursion under a `not`.

    Returns:
        A MongoDB query document, or None when nothing pushable could be spelled.
    """
    e = ir.get("e")
    const = _const_bool(ir)
    if const is not None:
        # `{}` matches every document; `$nor` of it matches none.
        return {} if const else {"$nor": [{}]}
    if e == "is_null" and ir["input"].get("e") == "col":
        return {ir["input"]["name"]: None}
    if e == "is_not_null" and ir["input"].get("e") == "col":
        return {ir["input"]["name"]: {"$ne": None}}
    if e == "not":
        # `$not` is only legal inside a field's operator document, so a whole-expression
        # negation is `$nor` of one term. Exact: negating a widened operand would narrow.
        inner = to_mongo_filter(ir["input"], exact=True)
        return None if inner is None else {"$nor": [inner]}
    if e == "in_list":
        parsed = _in_list(ir)
        if parsed is None:
            return None
        column, members = parsed
        return {column: {"$in": [_literal({"value": m}) for m in members]}}
    if e == "str":
        return _mongo_str(ir)
    if e != "binary":
        return None
    op = ir["op"]
    if op in ("and", "or"):
        left = to_mongo_filter(ir["left"], exact=exact)
        right = to_mongo_filter(ir["right"], exact=exact)
        both = {f"${op}": [left, right]} if left is not None and right is not None else None
        return both if exact else _combine(op, left, right, both)
    if op in COMPARISON_OPS:
        parsed = _col_and_literal(ir["left"], ir["right"])
        if parsed is None:
            return None
        col, value, flipped = parsed
        effective = COMPARISON_FLIP[op] if flipped else op
        return {col: {_MONGO_OP[effective]: value}}
    return None
