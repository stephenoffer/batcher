"""IR to the native parquet reader's compact predicate, for row-group pruning in Rust.

Kept in lockstep with `crates/bc-io/src/predicate.rs`: the tags here are that enum's serde
spelling, and an unrecognized one makes the reader reject the whole predicate silently.
That fixed vocabulary is why `IN` and `NOT` are *expressed* in it rather than added to it.
"""

from __future__ import annotations

from typing import Any

from batcher.io.predicate._shapes import _in_list
from batcher.plan.ir_tags import COMPARISON_FLIP, COMPARISON_OPS

__all__ = ["to_native_predicate"]

#: Longest `IN` list expanded into the native reader's OR-of-equalities. Kept far shorter
#: than the SQL cap because this one builds an actual predicate *tree* the row-group
#: pruner walks per row group, rather than a string the server parses once.
_NATIVE_IN_MAX = 64


def _native_scalar(ir: dict[str, Any]) -> tuple[Any, bool]:
    """A literal for the native reader: ``(value, ok)``.

    Only plain ``int``/``float``/``str``/``bool`` literals push to the native reader's
    zone-map pruning. Temporal kinds (``date``/``timestamp``/``time``) are epoch offsets
    whose parquet physical unit the reader cannot verify without risking an unsound prune,
    so they mark the term non-pushable (``ok=False``) and the pyarrow path handles them.
    """
    ((kind, value),) = ir["value"].items()
    if kind in ("int", "float", "str", "bool"):
        return value, True
    return None, False


#: The inverse of each comparison, used to carry a `NOT` down to the leaves.
#:
#: Sound on the reader's three-valued rows for the same reason it is in SQL: a null
#: operand makes both the operator and its inverse null, so neither the predicate nor its
#: negation ever selects that row, and the engine's `Filter` agrees.
_COMPARISON_NEGATE = {"eq": "ne", "ne": "eq", "lt": "ge", "ge": "lt", "gt": "le", "le": "gt"}


def _native_in_list(ir: dict[str, Any], negated: bool) -> dict[str, Any] | None:
    """An `IN` list as the reader's OR-of-equalities, or None.

    The native `Pred` enum has no set-membership node, and giving it one would be a wire
    change across the FFI for something the existing vocabulary already expresses exactly:
    ``a IN (1, 2, 3)`` is ``a = 1 OR a = 2 OR a = 3``, and the row-group pruner reaches the
    same verdict from either spelling. Negated, De Morgan makes it an `AND` of ``!=``.

    Capped at `_NATIVE_IN_MAX` members because this expands to a predicate *tree* the
    pruner walks per row group, so a long list would cost more to evaluate than the I/O it
    saves. A longer list declines and the engine filters.
    """
    parsed = _in_list(ir)
    if parsed is None:
        return None
    column, members = parsed
    if len(members) > _NATIVE_IN_MAX:
        return None
    terms: list[dict[str, Any]] = []
    for member in members:
        value, ok = _native_scalar({"value": member})
        if not ok:
            return None
        terms.append({"node": "cmp", "col": column, "op": "ne" if negated else "eq", "lit": value})
    node = "and" if negated else "or"
    folded = terms[0]
    for term in terms[1:]:
        folded = {"node": node, "left": folded, "right": term}
    return folded


def to_native_predicate(ir: dict[str, Any], *, negated: bool = False) -> dict[str, Any] | None:
    """Translate the pushable subset of `ir` to the native reader's compact predicate.

    The shape `bc_io`'s `predicate` module deserializes: ``{"node":"cmp","col":..,"op":..,
    "lit":..}`` / ``{"node":"and"/"or","left":..,"right":..}`` / ``{"node":"is_null","col":..,
    "negated":..}``. Comparisons are normalized so the column is on the left. Returns
    ``None`` if any term is not pushable (a non-column/literal comparison, a temporal
    literal, or an unsupported node) — the caller then reads without native pruning.

    The ``"is_null"`` tag is load-bearing: `bc_io`'s `Pred` is
    ``#[serde(tag = "node", rename_all = "snake_case")]``, so its `IsNull` variant is spelled
    ``is_null``. Emitting anything else makes `parse()` reject the *whole* predicate — and
    because pruning is only ever an optimization, that failure is silent (correct results, zero
    row-groups pruned). Keep this in lockstep with `crates/bc-io/src/predicate.rs`.

    `IN` lists and `NOT` are expressed in that same fixed vocabulary rather than added to it:
    a set becomes an ``OR`` of equalities and a negation is carried to the leaves by De
    Morgan. Both are exact, so neither needs the `exact` flag the other translators thread —
    this one already declines a partial `AND`.

    Args:
        ir: The predicate's IR dictionary.
        negated: Translate the negation of `ir`. Set by the recursion when it passes a
            `not`; callers leave it alone.

    Returns:
        The reader's predicate dictionary, or None when any term is unpushable.
    """
    e = ir.get("e")
    if e in ("is_null", "is_not_null"):
        inner = ir["input"]
        if inner.get("e") != "col":
            return None
        negated_null = (e == "is_not_null") != negated
        return {"node": "is_null", "col": inner["name"], "negated": negated_null}
    if e == "not":
        return to_native_predicate(ir["input"], negated=not negated)
    if e == "in_list":
        return _native_in_list(ir, negated)
    if e != "binary":
        return None
    op = ir["op"]
    if op in ("and", "or"):
        # De Morgan: the reader's vocabulary has no `not` node, so a negation is carried
        # down to the leaves, where every operator it can reach has an inverse already in
        # the vocabulary. This is exact at every step, which is what a negation requires.
        folded = {"and": "or", "or": "and"}[op] if negated else op
        left = to_native_predicate(ir["left"], negated=negated)
        right = to_native_predicate(ir["right"], negated=negated)
        if left is None or right is None:
            return None
        return {"node": folded, "left": left, "right": right}
    if op in COMPARISON_OPS:
        left, right = ir["left"], ir["right"]
        if left.get("e") == "col" and right.get("e") == "lit":
            col, lit_ir, flipped = left["name"], right, False
        elif left.get("e") == "lit" and right.get("e") == "col":
            col, lit_ir, flipped = right["name"], left, True
        else:
            return None
        value, ok = _native_scalar(lit_ir)
        if not ok:
            return None
        effective = COMPARISON_FLIP[op] if flipped else op
        return {
            "node": "cmp",
            "col": col,
            "op": _COMPARISON_NEGATE[effective] if negated else effective,
            "lit": value,
        }
    return None
