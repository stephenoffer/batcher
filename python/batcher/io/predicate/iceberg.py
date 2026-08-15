"""IR to a `pyiceberg` row filter, for Iceberg scans and ``replace_where``.

The one translator with two callers wanting different answers: a scan may widen, because
its filter only prunes what is read, while `replace_where` chooses the rows to *overwrite*
and must not. `allow_partial` is that distinction.
"""

from __future__ import annotations

from typing import Any

from batcher.io.predicate._literals import _col_and_literal, _literal
from batcher.io.predicate._shapes import _combine, _const_bool, _in_list, _str_predicate
from batcher.plan.ir_tags import COMPARISON_FLIP, COMPARISON_OPS

__all__ = ["to_iceberg_expression"]


def to_iceberg_expression(ir: dict[str, Any], *, allow_partial: bool = False) -> Any | None:
    """Translate the pushable subset of `ir` to a `pyiceberg` row filter, or None.

    `allow_partial` lets an `AND` push whichever conjuncts translated, which is what a
    *scan* wants: the filter only prunes I/O and the engine's `Filter` re-checks the rows.
    It defaults off because the same translation drives ``replace_where``, where a widened
    predicate would overwrite rows the caller never named. A caller pruning a read opts in;
    a caller choosing rows to replace must not.

    Args:
        ir: The predicate's IR dictionary.
        allow_partial: Push the translatable conjuncts of an `AND` instead of declining
            the whole expression. Only ever correct for a read.

    Returns:
        The `pyiceberg` expression, or None when nothing translates.
    """
    from pyiceberg import expressions as ie

    cmp_ctor = {
        "eq": ie.EqualTo,
        "ne": ie.NotEqualTo,
        "lt": ie.LessThan,
        "le": ie.LessThanOrEqual,
        "gt": ie.GreaterThan,
        "ge": ie.GreaterThanOrEqual,
    }

    def walk(node: dict[str, Any], exact: bool = not allow_partial) -> Any | None:
        e = node.get("e")
        const = _const_bool(node)
        if const is not None:
            return ie.AlwaysTrue() if const else ie.AlwaysFalse()
        if e == "is_null" and node["input"].get("e") == "col":
            return ie.IsNull(node["input"]["name"])
        if e == "is_not_null" and node["input"].get("e") == "col":
            return ie.NotNull(node["input"]["name"])
        if e == "not":
            # Exact: negating a partially-translated operand narrows the filter, which
            # would drop rows on a scan and delete unnamed rows under `replace_where`.
            inner = walk(node["input"], True)
            return None if inner is None else ie.Not(inner)
        if e == "in_list":
            parsed = _in_list(node)
            if parsed is None:
                return None
            column, members = parsed
            return ie.In(column, [_literal({"value": m}) for m in members])
        if e == "str":
            parsed_str = _str_predicate(node)
            # Iceberg models the prefix case only; the manifest metadata it prunes with
            # (lower/upper bounds per column) cannot answer a suffix or an infix.
            if parsed_str is None or parsed_str[1] != "starts_with":
                return None
            return ie.StartsWith(parsed_str[0], parsed_str[2])
        if e != "binary":
            return None
        op = node["op"]
        if op in ("and", "or"):
            left = walk(node["left"], exact)
            right = walk(node["right"], exact)
            if left is None or right is None:
                return None if exact else _combine(op, left, right, None)
            return ie.And(left, right) if op == "and" else ie.Or(left, right)
        if op in COMPARISON_OPS:
            parsed = _col_and_literal(node["left"], node["right"])
            if parsed is None:
                return None
            col, value, flipped = parsed
            effective = COMPARISON_FLIP[op] if flipped else op
            return cmp_ctor[effective](col, value)
        return None

    return walk(ir)
