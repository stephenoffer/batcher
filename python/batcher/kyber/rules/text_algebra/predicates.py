"""A string call whose *comparison* is the real predicate, restated as that predicate.

Three shapes, all of them what a SQL author writes when the direct spelling does not come
to mind or does not exist in the dialect they came from:

* `position(s, 'x') > 0` and `regexp_count(s, p) > 0` are membership tests written as
  arithmetic. Both functions return a count, and every comparison of that count against
  zero is a `contains` / `regexp_matches` in disguise. The direct form is cheaper — a
  search that stops at the first hit instead of counting every one — and, unlike a
  comparison against a computed count, it is a shape the later rules recognize.
* `substr(s, 1, 3) = 'abc'` is `starts_with(s, 'abc')`, and `right(s, 3) = 'abc'` is
  `ends_with(s, 'abc')`. This is the biggest win in the module, because
  `like_prefix_to_range` then turns the prefix test into the byte range a source can push
  into a scan, and a leading `substr` can push nothing.
* `reverse(s) = 'cba'` is `s = 'abc'`. Reversal is an involution and injective, so the
  equality moves onto the bare column and the call disappears entirely.

The length equalities are the guard that keeps the `substr`/`right` rules honest. Taking
three characters from a two-character string yields the whole string, which cannot equal a
three-character literal — and `starts_with` answers `false` for that row too, so the two
agree. They would *not* agree if the literal's length differed from the requested count,
which is why the rules compare the two and decline when they differ. The literal is also
required to be ASCII, so a Python character count matches the engine's.
"""

from __future__ import annotations

from collections.abc import Callable

from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rule import Phase, node_rule
from batcher.kyber.rules.leaf_rewrite import rewrite_node
from batcher.plan.expr_ir import Binary, Expr, Lit, Not
from batcher.plan.expr_ir.func_nodes import StrFunc
from batcher.plan.logical import Aggregate, Filter, Project, Sort, Window

__all__ = ["COUNTING_PREDICATE_RULES", "REVERSE_EQUALITY_RULES", "SLICE_PREDICATE_RULES"]

_NODES = (Filter, Project, Aggregate, Sort, Window)
_FLIP = {"eq": "eq", "ne": "ne", "lt": "gt", "gt": "lt", "le": "ge", "ge": "le"}


def _int_literal(expr: Expr) -> int | None:
    if isinstance(expr, Lit) and isinstance(expr.value, int) and not isinstance(expr.value, bool):
        return expr.value
    return None


def _comparison(expr: Expr) -> tuple[str, Expr, Expr] | None:
    """`(op, computed_side, literal_side)` with the operator normalized to the left."""
    if not isinstance(expr, Binary) or expr.op not in _FLIP:
        return None
    if isinstance(expr.right, Lit):
        return expr.op, expr.left, expr.right
    if isinstance(expr.left, Lit):
        return _FLIP[expr.op], expr.right, expr.left
    return None


# --- a count compared against zero is a membership test ----------------------

#: `(counting function, comparison against zero) -> (membership function, negated)`.
#: `position` answers 0 when the needle is absent and a 1-based index otherwise;
#: `regexp_count` answers 0 when nothing matches. So "positive" is "present" in both, in
#: each of the three spellings a query uses for it.
_COUNTING: dict[str, str] = {"position": "contains", "regexp_count": "regexp_matches"}
_PRESENCE = {("gt", 0): False, ("ge", 1): False, ("ne", 0): False, ("eq", 0): True}


def _counting_leaf(fn: str, key: tuple[str, int]) -> Callable[[Expr], Expr]:
    negated = _PRESENCE[key]

    def leaf(expr: Expr) -> Expr:
        parts = _comparison(expr)
        if parts is None:
            return expr
        op, computed, literal = parts
        if (op, _int_literal(literal)) != key:
            return expr
        if not isinstance(computed, StrFunc) or computed.fn != fn:
            return expr
        if not isinstance(computed.pattern, str):
            return expr
        membership = StrFunc(_COUNTING[fn], computed.input, pattern=computed.pattern)
        return Not(membership) if negated else membership

    return leaf


#: The mirror of each comparison, for the operator index: `_comparison` normalizes the
#: computed side to the left, so a leaf built for `op` is also reached by `_FLIP[op]` nodes.
_FLIP = {"eq": "eq", "ne": "ne", "lt": "gt", "gt": "lt", "le": "ge", "ge": "le"}


def _register(name: str, leaf: Callable[[Expr], Expr], op: str):
    return DEFAULT_REGISTRY.add(
        node_rule(
            name,
            Phase.NORMALIZE,
            lambda node, _ctx, _leaf=leaf: rewrite_node(node, _leaf),
            matches=_NODES,
            expr_fn=leaf,
            expr_matches=(Binary,),
            expr_ops=(op, _FLIP[op]),
        )
    )


_COUNT_SUFFIX = {
    ("gt", 0): "gt_zero",
    ("ge", 1): "ge_one",
    ("ne", 0): "ne_zero",
    ("eq", 0): "eq_zero",
}


#: `position(s, p) > 0`, `>= 1`, `<> 0` -> `contains(s, p)`, and `= 0` -> `NOT contains(…)`;
#: the same four for `regexp_count(s, p)` against `regexp_matches(s, p)`. Eight rules, one
#: per (function, comparison) pair, so a plan that only ever writes one of the spellings
#: pays for one rule rather than a dispatch table.
COUNTING_PREDICATE_RULES = [
    _register(f"{fn}_{_COUNT_SUFFIX[key]}_to_membership", _counting_leaf(fn, key), key[0])
    for fn in _COUNTING
    for key in _PRESENCE
]


# --- a compared slice is a prefix or suffix test -----------------------------


def _ascii_literal(expr: Expr) -> str | None:
    """The value of an ASCII string literal, else ``None``.

    ASCII is required so that Python's character count is the engine's character count;
    a multi-byte literal would make the length comparison below meaningless.
    """
    if isinstance(expr, Lit) and isinstance(expr.value, str) and expr.value.isascii():
        return expr.value
    return None


def _slice_leaf(fn: str, *, negated: bool) -> Callable[[Expr], Expr]:
    """`substr(s, 1, n) = 'lit'` -> `starts_with(s, 'lit')`, and the `right`/`ends_with`
    and `<>` variants, when `n` equals the literal's length."""
    target = "starts_with" if fn == "substr" else "ends_with"
    op_wanted = "ne" if negated else "eq"

    def leaf(expr: Expr) -> Expr:
        parts = _comparison(expr)
        if parts is None:
            return expr
        op, computed, literal = parts
        if op != op_wanted or not isinstance(computed, StrFunc) or computed.fn != fn:
            return expr
        value = _ascii_literal(literal)
        if value is None:
            return expr
        if fn == "substr":
            if computed.start != 1 or computed.length != len(value):
                return expr
        elif computed.start != len(value) or computed.length is not None:
            return expr
        test = StrFunc(target, computed.input, pattern=value)
        return Not(test) if negated else test

    return leaf


#: `substr(s, 1, n) = 'lit'` -> `starts_with(s, 'lit')` and `right(s, n) = 'lit'` ->
#: `ends_with(s, 'lit')`, each with its `<>` counterpart, whenever `n` is exactly the
#: literal's length. A slice of a different length is a different question and is declined.
SLICE_PREDICATE_RULES = [
    _register(
        f"{fn}_{'ne' if negated else 'eq'}_literal_to_affix_test",
        _slice_leaf(fn, negated=negated),
        "ne" if negated else "eq",
    )
    for fn in ("substr", "right")
    for negated in (False, True)
]


# --- reversal is injective, so the comparison moves onto the column ----------


def _reverse_leaf(op_wanted: str) -> Callable[[Expr], Expr]:
    def leaf(expr: Expr) -> Expr:
        parts = _comparison(expr)
        if parts is None:
            return expr
        op, computed, literal = parts
        if op != op_wanted or not isinstance(computed, StrFunc) or computed.fn != "reverse":
            return expr
        value = _ascii_literal(literal)
        if value is None:
            return expr
        return Binary(op, computed.input, Lit(value[::-1]))

    return leaf


#: `reverse(s) = 'cba'` -> `s = 'abc'`, and the `<>` dual. Reversal is a bijection on
#: strings, so the comparison is exactly as selective either way — but on the bare column
#: it is sargable, and the per-row reversal disappears.
REVERSE_EQUALITY_RULES = [
    _register(f"reverse_{op}_literal_to_column_{op}", _reverse_leaf(op), op) for op in ("eq", "ne")
]
