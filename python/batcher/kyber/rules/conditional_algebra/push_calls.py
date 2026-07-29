"""Push a scalar call through a `CASE` onto each of its branch values.

`f(CASE WHEN c THEN a ELSE b END)` and `CASE WHEN c THEN f(a) ELSE f(b) END` compute the
same thing for every deterministic `f`: the `CASE` decides *which* value is produced, and
`f` transforms whichever one that is. Moving `f` inside changes nothing about the
selection, and the conditions are evaluated exactly once in either form.

The point is what the pushed form enables. A `CASE`'s branch values are overwhelmingly
literals — a status code mapped to a label, a null-safe default, a unit conversion — and
`f(literal)` is a constant that `fold_case_of_literal` then collapses. Once every branch is
a boolean constant, `case_boolean_branches_to_predicate` reduces the whole expression to
the branch condition. So this family's real output is not a cheaper call; it is the
disappearance of the `CASE`.

Each family is a separate rule, matched by node type, so a plan that only ever wraps a
`CASE` in a string function pays for one rule rather than a dispatch over all of them. The
vocabularies are the null-strict, total ones from `nulls/strictness`: pushing a call that
can *raise* would move the error from one row set to another, since a vectorized `CASE`
evaluates every branch and a pushed call evaluates only the branch it lands on.
"""

from __future__ import annotations

from collections.abc import Callable

from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rule import Phase, node_rule
from batcher.kyber.rules.leaf_rewrite import rewrite_node
from batcher.kyber.rules.nulls.strictness import STRICT_MATH_FNS, STRICT_STR_FNS
from batcher.plan.expr_ir import Case, Expr, InList
from batcher.plan.expr_ir.core import Binary, IsInf, IsNan, MathExpr
from batcher.plan.expr_ir.func_nodes import (
    ConvertTimezone,
    DateFunc,
    DateOffset,
    DateTrunc,
    ListContains,
    ListFunc,
    ListSlice,
    Strftime,
    StrFunc,
    Strptime,
    StructField,
)
from batcher.plan.logical import Aggregate, Filter, Project, Sort, Window

__all__ = ["CASE_PUSH_RULES"]

_NODES = (Filter, Project, Aggregate, Sort, Window)


def _rebuild_unary(call: Expr, value: Expr) -> Expr:
    """`call` with its single operand replaced by `value`, preserving every other slot."""
    if isinstance(call, MathExpr):
        return MathExpr(call.fn, value)
    if isinstance(call, StrFunc):
        return StrFunc(
            call.fn,
            value,
            pattern=call.pattern,
            replacement=call.replacement,
            start=call.start,
            length=call.length,
        )
    if isinstance(call, DateFunc):
        return DateFunc(call.fn, value)
    if isinstance(call, DateTrunc):
        return DateTrunc(value, call.unit)
    if isinstance(call, Strftime):
        return Strftime(value, call.format)
    if isinstance(call, Strptime):
        return Strptime(value, call.format)
    if isinstance(call, ConvertTimezone):
        return ConvertTimezone(value, call.from_tz, call.to_tz)
    if isinstance(call, DateOffset):
        return DateOffset(value, call.months, call.days, call.micros)
    if isinstance(call, ListFunc):
        return ListFunc(call.fn, value)
    if isinstance(call, ListSlice):
        return ListSlice(value, call.offset, call.length)
    if isinstance(call, ListContains):
        return ListContains(value, call.value)
    if isinstance(call, StructField):
        return StructField(value, call.field)
    if isinstance(call, InList):
        return InList(value, call.values)
    if isinstance(call, IsNan):
        return IsNan(value)
    return IsInf(value)


def _push_unary(matches: Callable[[Expr], bool]) -> Callable[[Expr], Expr]:
    """The leaf rewrite pushing a unary call the `matches` predicate accepts."""

    def leaf(expr: Expr) -> Expr:
        if not matches(expr):
            return expr
        case = getattr(expr, "input", None)
        if not isinstance(case, Case):
            return expr
        return Case(
            [(cond, _rebuild_unary(expr, value)) for cond, value in case.branches],
            _rebuild_unary(expr, case.otherwise),
        )

    return leaf


def _push_binary(ops: frozenset[str]) -> Callable[[Expr], Expr]:
    """The leaf rewrite pushing a binary operator onto each branch of a `CASE` operand.

    Only one side may be the `CASE`; the other is duplicated into every branch, which is
    why it is restricted to the total operators. Whichever side carries the `CASE`, the
    operand order is preserved so a non-commutative operator keeps its meaning.
    """

    def leaf(expr: Expr) -> Expr:
        if not isinstance(expr, Binary) or expr.op not in ops:
            return expr
        if isinstance(expr.left, Case) and not isinstance(expr.right, Case):
            case, other, case_on_left = expr.left, expr.right, True
        elif isinstance(expr.right, Case) and not isinstance(expr.left, Case):
            case, other, case_on_left = expr.right, expr.left, False
        else:
            return expr

        def apply(value: Expr) -> Expr:
            return Binary(expr.op, value, other) if case_on_left else Binary(expr.op, other, value)

        return Case([(cond, apply(value)) for cond, value in case.branches], apply(case.otherwise))

    return leaf


def _register(name: str, leaf: Callable[[Expr], Expr], expr_matches: tuple[type, ...]):
    return DEFAULT_REGISTRY.add(
        node_rule(
            name,
            Phase.NORMALIZE,
            lambda node, _ctx, _leaf=leaf: rewrite_node(node, _leaf),
            matches=_NODES,
            expr_fn=leaf,
            expr_matches=expr_matches,
        )
    )


#: `(rule suffix, node predicate)` for each unary family pushed into the branches. Every
#: vocabulary is the null-strict, total one — a call that can raise is not pushed.
#: Each entry also names the `Expr` type its predicate accepts, so the driver can offer the
#: rule only to expressions of that type — the type is exactly what the predicate tests.
_UNARY_FAMILIES: tuple[tuple[str, Callable[[Expr], bool], type], ...] = (
    ("math_fn", lambda e: isinstance(e, MathExpr) and e.fn in STRICT_MATH_FNS, MathExpr),
    ("str_fn", lambda e: isinstance(e, StrFunc) and e.fn in STRICT_STR_FNS, StrFunc),
    ("date_fn", lambda e: isinstance(e, DateFunc), DateFunc),
    ("date_trunc", lambda e: isinstance(e, DateTrunc), DateTrunc),
    ("strftime", lambda e: isinstance(e, Strftime), Strftime),
    ("nan_check", lambda e: isinstance(e, IsNan), IsNan),
    ("inf_check", lambda e: isinstance(e, IsInf), IsInf),
    ("strptime", lambda e: isinstance(e, Strptime), Strptime),
    ("convert_timezone", lambda e: isinstance(e, ConvertTimezone), ConvertTimezone),
    ("date_offset", lambda e: isinstance(e, DateOffset), DateOffset),
    ("list_reduction", lambda e: isinstance(e, ListFunc), ListFunc),
    ("list_slice", lambda e: isinstance(e, ListSlice), ListSlice),
    ("list_contains", lambda e: isinstance(e, ListContains), ListContains),
    ("struct_field", lambda e: isinstance(e, StructField), StructField),
    ("in_list", lambda e: isinstance(e, InList), InList),
)

#: `(rule suffix, operator vocabulary)` for each binary family pushed into the branches.
_BINARY_FAMILIES: tuple[tuple[str, frozenset[str]], ...] = (
    ("and", frozenset({"and"})),
    ("or", frozenset({"or"})),
    ("bit_and", frozenset({"bit_and"})),
    ("bit_or", frozenset({"bit_or"})),
    ("bit_xor", frozenset({"bit_xor"})),
)


#: Twenty rules pushing a call into a `CASE`'s branches: fifteen unary families (the math
#: and string functions, the date part / `date_trunc` / `strftime` / `strptime` /
#: `convert_timezone` / `offset_by` temporal calls, the NaN and infinity predicates, the
#: list reduction / slice / membership accessors, a struct field, and an `IN` list) and five
#: binary ones (the two Kleene connectives and the three bitwise operators). Together with
#: the arithmetic and comparison pushes in `extra/conditional`, every total scalar operation
#: the engine has can now reach a `CASE`'s branch literals and fold there.
CASE_PUSH_RULES = [
    _register(f"push_{suffix}_into_case_branches", _push_unary(matches), (node_type,))
    for suffix, matches, node_type in _UNARY_FAMILIES
] + [
    _register(f"push_{suffix}_into_case_branches", _push_binary(ops), (Binary,))
    for suffix, ops in _BINARY_FAMILIES
]
