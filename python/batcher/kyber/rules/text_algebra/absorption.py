"""Two string predicates on one column where one implies the other.

`p AND q` keeps only the **stronger** predicate and `p OR q` keeps only the **weaker**
one, exactly as boolean absorption does for `p AND (p OR q)`. What makes it a separate
family is that the implication is decided by a string relation rather than by syntactic
equality: `starts_with(s, 'a/b/')` implies `starts_with(s, 'a/')` because one pattern is a
prefix of the other, and no amount of boolean algebra can see that.

The shape is common because it is generated rather than written. A `LIKE 'a/%'` from a
partition filter and a `LIKE 'a/b/%'` from a user predicate meet at the same node after
pushdown, and `like_prefix_to_starts_with` turns both into `starts_with` calls before this
family looks at them. Dropping the weaker one removes a whole pass over the column, and —
more importantly — leaves a single `starts_with` that `like_prefix_to_range` can turn into
the byte range a source can push into a scan.

Every rule here is exact under three-valued logic. If `p` implies `q` on every non-null
row and both are null-strict on the same operand, then `p AND q` and `p` agree on `true`,
on `false`, and on `NULL` — the null row makes both sides `NULL` rather than making the
weaker conjunct decide. That symmetry is why these need no non-null guard, and it is the
reason the implication is required to hold for the *pattern*, never for the data.
"""

from __future__ import annotations

from collections.abc import Callable

from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rule import Phase, node_rule
from batcher.kyber.rules.leaf_rewrite import rewrite_node
from batcher.plan.expr_ir import Binary, Expr, Lit
from batcher.plan.expr_ir.func_nodes import StrFunc
from batcher.plan.expr_rewrite import expr_key
from batcher.plan.logical import Aggregate, Filter, Project, Sort, Window

__all__ = ["STRING_ABSORPTION_RULES"]

_NODES = (Filter, Project, Aggregate, Sort, Window)

#: For each matching predicate family, the relation "`strong` implies `weak`" between two
#: patterns. A prefix predicate is stronger the longer its pattern is; a substring
#: predicate is stronger the longer the substring it demands.
_IMPLIES: dict[str, Callable[[str, str], bool]] = {
    "starts_with": lambda strong, weak: strong.startswith(weak),
    "ends_with": lambda strong, weak: strong.endswith(weak),
    "contains": lambda strong, weak: weak in strong,
}


def _pattern_call(expr: Expr, fn: str) -> tuple[Expr, str] | None:
    """`(operand, pattern)` for a `fn(operand, pattern)` call with a string pattern."""
    if isinstance(expr, StrFunc) and expr.fn == fn and isinstance(expr.pattern, str):
        return expr.input, expr.pattern
    return None


def _string_equality(expr: Expr) -> tuple[Expr, str] | None:
    """`(operand, value)` for an `operand = 'literal'` comparison against a string."""
    if not isinstance(expr, Binary) or expr.op != "eq":
        return None
    for operand, other in ((expr.left, expr.right), (expr.right, expr.left)):
        if isinstance(other, Lit) and isinstance(other.value, str):
            return operand, other.value
    return None


def _absorb_pattern_pair(expr: Expr, fn: str, *, conjunction: bool) -> Expr:
    """Drop the redundant side of `p AND q` / `p OR q` for two `fn` calls on one column.

    A conjunction keeps the stronger predicate, a disjunction the weaker one.
    """
    connective = "and" if conjunction else "or"
    if not isinstance(expr, Binary) or expr.op != connective:
        return expr
    left, right = _pattern_call(expr.left, fn), _pattern_call(expr.right, fn)
    if left is None or right is None:
        return expr
    (left_operand, left_pattern), (right_operand, right_pattern) = left, right
    if expr_key(left_operand) != expr_key(right_operand) or left_pattern == right_pattern:
        return expr
    implies = _IMPLIES[fn]
    if implies(left_pattern, right_pattern):
        strong, weak = expr.left, expr.right
    elif implies(right_pattern, left_pattern):
        strong, weak = expr.right, expr.left
    else:
        return expr
    return strong if conjunction else weak


def _absorb_equality(expr: Expr, *, conjunction: bool) -> Expr:
    """Drop the redundant side of `x = 'lit' AND fn(x, p)` / the `OR` dual.

    An equality is the strongest predicate there is, so it wins a conjunction outright —
    provided the literal actually satisfies the pattern predicate, which is the whole
    condition being checked.
    """
    connective = "and" if conjunction else "or"
    if not isinstance(expr, Binary) or expr.op != connective:
        return expr
    for equality, other in ((expr.left, expr.right), (expr.right, expr.left)):
        eq = _string_equality(equality)
        if eq is None:
            continue
        operand, value = eq
        for fn, implies in _IMPLIES.items():
            call = _pattern_call(other, fn)
            if call is None:
                continue
            call_operand, pattern = call
            if expr_key(call_operand) != expr_key(operand):
                continue
            if implies(value, pattern):
                return equality if conjunction else other
    return expr


def _pattern_leaf(fn: str, *, conjunction: bool) -> Callable[[Expr], Expr]:
    def leaf(expr: Expr) -> Expr:
        return _absorb_pattern_pair(expr, fn, conjunction=conjunction)

    return leaf


def _equality_leaf(*, conjunction: bool) -> Callable[[Expr], Expr]:
    def leaf(expr: Expr) -> Expr:
        return _absorb_equality(expr, conjunction=conjunction)

    return leaf


def _register(name: str, leaf: Callable[[Expr], Expr], connective: str):
    return DEFAULT_REGISTRY.add(
        node_rule(
            name,
            Phase.NORMALIZE,
            lambda node, _ctx, _leaf=leaf: rewrite_node(node, _leaf),
            matches=_NODES,
            expr_fn=leaf,
            expr_matches=(Binary,),
            # Each rule is built for one connective and returns its input for the other, so
            # the index names it exactly rather than the whole `Binary` type.
            expr_ops=(connective,),
        )
    )


_NAMES = {
    ("starts_with", True): "absorb_weaker_prefix_conjunct",
    ("starts_with", False): "absorb_stronger_prefix_disjunct",
    ("ends_with", True): "absorb_weaker_suffix_conjunct",
    ("ends_with", False): "absorb_stronger_suffix_disjunct",
    ("contains", True): "absorb_weaker_substring_conjunct",
    ("contains", False): "absorb_stronger_substring_disjunct",
}


#: Eight absorption rules over the string predicates:
#:
#:   * `starts_with(s, 'a/') AND starts_with(s, 'a/b/')` -> the longer prefix, and the
#:     `OR` dual keeping the shorter one; likewise for `ends_with` on suffixes and for
#:     `contains` on substrings.
#:   * `s = 'abc' AND starts_with(s, 'ab')` -> the equality, and the `OR` dual keeping the
#:     pattern predicate. The equality is checked against all three pattern families, so
#:     one rule covers prefix, suffix and substring.
#:
#: Each is a fixpoint on its own output: the surviving side is a single predicate, not a
#: connective, so the rule cannot re-fire on what it produced.
STRING_ABSORPTION_RULES = [
    _register(
        _NAMES[(fn, conjunction)],
        _pattern_leaf(fn, conjunction=conjunction),
        "and" if conjunction else "or",
    )
    for fn in _IMPLIES
    for conjunction in (True, False)
] + [
    _register(name, _equality_leaf(conjunction=conjunction), "and" if conjunction else "or")
    for name, conjunction in (
        ("absorb_string_equality_conjunct", True),
        ("absorb_string_equality_disjunct", False),
    )
]
