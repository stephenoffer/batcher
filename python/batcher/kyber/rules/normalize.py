"""NORMALIZE-phase whole-tree rewrites — constant folding and expression simplification.

These two confluent rewrites run over the whole plan (every node's expressions) in
the NORMALIZE phase, before the local algebraic rules and the pushdown phase.

Constant folding evaluates constant sub-expressions at plan time: `col("x") > 2 + 3`
becomes `col("x") > 5`; `1 == 1` becomes `true`. Correctness is anchored to the
engine, not Python — we only fold where Python's result is **bit-identical** to what
`bc-expr` would compute (so comparisons and boolean ops fold freely, but integer
division/modulo and mixed int/float arithmetic are left alone).

Expression simplification then drops algebraic identity operations a query (or the
prior fold) leaves behind: `x AND true → x`, `x OR false → x`, `x + 0 → x`,
`x * 1 → x`, `NOT NOT x → x`. Only **identity-element** rewrites are applied, never
annihilators (the engine's boolean ops are non-Kleene), and numeric identities use
*integer* `0`/`1` only.

Both are registered as `plan_rule`s in `Phase.NORMALIZE` (see
`kyber.registry.register_builtin_rules`). Their pure functions (`fold_constants`,
`simplify_expressions`) stay importable for unit tests.
"""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import DEFAULT_REGISTRY, rule
from batcher.kyber.rule import Phase, plan_rule
from batcher.plan.expr_ir import Binary, Cast, Col, Expr, Lit, Not
from batcher.plan.expr_ir.namespaces import StrFunc
from batcher.plan.expr_rewrite import (
    combine_conjuncts,
    map_node_expressions,
    split_conjuncts,
    transform_expr_up,
)
from batcher.plan.logical import Filter, LogicalPlan
from batcher.plan.visitor import transform_up

__all__ = [
    "ConstantFolding",
    "ExprSimplification",
    "fold_constants",
    "like_prefix_to_range",
    "or_to_in_and_range",
    "simplify_expressions",
]

_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1
_COMPARISONS = {"gt": ">", "ge": ">=", "lt": "<", "le": "<=", "eq": "==", "ne": "!="}


# --- Constant folding -------------------------------------------------------


def fold_constants(plan: LogicalPlan) -> LogicalPlan:
    """Fold constant sub-expressions throughout the plan."""
    return transform_up(plan, lambda n: map_node_expressions(n, _fold_expr))


def _fold_expr(expr: Expr) -> Expr:
    return transform_expr_up(expr, _fold)


def _fold(expr: Expr) -> Expr:
    """Leaf rule: fold a node whose children are already folded."""
    if isinstance(expr, Not) and _is_bool(expr.input):
        return Lit(not expr.input.value)
    if isinstance(expr, Binary) and isinstance(expr.left, Lit) and isinstance(expr.right, Lit):
        folded = _fold_binary(expr.op, expr.left.value, expr.right.value)
        if folded is not None:
            return folded
    return expr


def _fold_binary(op: str, a: object, b: object) -> Lit | None:
    if op in _COMPARISONS:
        if not _comparable(a, b):
            return None
        return Lit(_compare(op, a, b))
    if op in ("and", "or"):
        if _is_bool_val(a) and _is_bool_val(b):
            return Lit(a and b) if op == "and" else Lit(a or b)
        return None
    return _fold_arith(op, a, b)


def _fold_arith(op: str, a: object, b: object) -> Lit | None:
    same_int = _is_int_val(a) and _is_int_val(b)
    same_float = isinstance(a, float) and isinstance(b, float)
    if not (same_int or same_float):
        return None  # mixed int/float: Arrow's promotion differs — don't fold
    if op == "add":
        r = a + b
    elif op == "sub":
        r = a - b
    elif op == "mul":
        r = a * b
    elif op == "div" and same_float:
        if b == 0.0:
            return None
        r = a / b
    else:
        # int div/mod (Arrow truncates ≠ Python), and float mod: leave alone.
        return None
    if same_int and not (_INT64_MIN <= r <= _INT64_MAX):
        return None  # would overflow int64 differently than Arrow
    return Lit(r)


def _compare(op: str, a: object, b: object) -> bool:
    return {
        "gt": a > b,
        "ge": a >= b,
        "lt": a < b,
        "le": a <= b,
        "eq": a == b,
        "ne": a != b,
    }[op]


def _comparable(a: object, b: object) -> bool:
    # Only same-kind comparisons (both numeric / both str / both bool) match the
    # engine; mixing kinds either errors there or has surprising semantics.
    if _is_bool_val(a) and _is_bool_val(b):
        return True
    if isinstance(a, str) and isinstance(b, str):
        return True
    return _is_number(a) and _is_number(b)


def _is_bool(expr: Expr) -> bool:
    return isinstance(expr, Lit) and isinstance(expr.value, bool)


def _is_bool_val(x: object) -> bool:
    return isinstance(x, bool)


def _is_int_val(x: object) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_number(x: object) -> bool:
    return (isinstance(x, (int, float))) and not isinstance(x, bool)


class ConstantFolding:
    """Pass: fold constant sub-expressions throughout the plan."""

    name = "constant_folding"

    def apply(self, plan: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan:
        return fold_constants(plan)


# --- Expression simplification ----------------------------------------------


def simplify_expressions(plan: LogicalPlan) -> LogicalPlan:
    """Apply identity simplifications throughout the plan."""
    return transform_up(plan, lambda n: map_node_expressions(n, _simplify_expr))


def _simplify_expr(expr: Expr) -> Expr:
    return transform_expr_up(expr, _simplify)


def _simplify(expr: Expr) -> Expr:
    if isinstance(expr, Not) and isinstance(expr.input, Not):
        return expr.input.input  # NOT NOT x → x

    # Cast(Cast(x, t), t) → Cast(x, t): casting to a type then to that same type again
    # is redundant. Only when the dtype AND try_cast semantics match — a strict cast
    # wrapping a try-cast (or vice versa) is not equivalent (different null behavior).
    if (
        isinstance(expr, Cast)
        and isinstance(expr.input, Cast)
        and expr.input.dtype == expr.dtype
        and expr.input.try_cast == expr.try_cast
    ):
        return Cast(expr.input.input, expr.dtype, try_cast=expr.try_cast)

    if not isinstance(expr, Binary):
        return expr
    op, left, right = expr.op, expr.left, expr.right

    if op == "and":
        if _is_true(right):
            return left
        if _is_true(left):
            return right
    elif op == "or":
        if _is_false(right):
            return left
        if _is_false(left):
            return right
    elif op == "add":
        if _is_int_zero(right):
            return left
        if _is_int_zero(left):
            return right
    elif op == "sub":
        if _is_int_zero(right):
            return left
    elif op == "mul":
        if _is_int_one(right):
            return left
        if _is_int_one(left):
            return right
    elif op == "div":
        if _is_int_one(right):
            return left
    return expr


def _is_true(expr: Expr) -> bool:
    return isinstance(expr, Lit) and expr.value is True


def _is_false(expr: Expr) -> bool:
    return isinstance(expr, Lit) and expr.value is False


def _is_int_zero(expr: Expr) -> bool:
    return isinstance(expr, Lit) and type(expr.value) is int and expr.value == 0


def _is_int_one(expr: Expr) -> bool:
    return isinstance(expr, Lit) and type(expr.value) is int and expr.value == 1


class ExprSimplification:
    """Pass: drop algebraic identity operations throughout the plan."""

    name = "expr_simplification"

    def apply(self, plan: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan:
        return simplify_expressions(plan)


# --- LIKE-prefix → range ----------------------------------------------------

# Characters that make a LIKE pattern more than a plain prefix.
_PATTERN_SPECIAL = frozenset("%_\\")
# Largest last-prefix character we will increment: the increment must stay a single
# byte whose UTF-8 order matches its code point (true for ASCII below 0x7F).
_MAX_INCREMENTABLE = 0x7E


def like_prefix_to_range(plan: LogicalPlan) -> LogicalPlan:
    """Rewrite every pure-prefix `LIKE 'abc%'` to the exact range `col >= 'abc' AND
    col < 'abd'`.

    With no other wildcards the range is exact — a string matches `'abc%'` iff it is
    in `['abc', 'abd')` — so the opaque `LIKE` is replaced by plain comparisons that
    zone-map pruning and predicate pushdown can use (both are blind to `LIKE`). The
    classic prefix-search accelerant DuckDB/Spark apply, feeding Batcher's
    metadata-driven `zonemap_prune_filter`. Conservative: it fires only for a
    `<prefix>%` pattern whose prefix is non-empty, contains no further `%`/`_`/escape,
    and ends in a safely-incrementable ASCII character; `ILIKE` and mid-string
    wildcards are left untouched.
    """
    return transform_up(plan, lambda node: map_node_expressions(node, _rewrite_like))


def _rewrite_like(expr: Expr) -> Expr:
    if not (isinstance(expr, StrFunc) and expr.fn == "like" and isinstance(expr.pattern, str)):
        return expr
    upper = _prefix_upper_bound(expr.pattern)
    if upper is None:
        return expr
    prefix = expr.pattern[:-1]
    return Binary(
        "and",
        Binary("ge", expr.input, Lit(prefix)),
        Binary("lt", expr.input, Lit(upper)),
    )


def _prefix_upper_bound(pattern: str) -> str | None:
    """The exclusive upper bound for `<prefix>%`, or None if not a safe pure prefix."""
    if len(pattern) < 2 or not pattern.endswith("%"):
        return None
    prefix = pattern[:-1]
    if any(c in _PATTERN_SPECIAL for c in prefix):
        return None
    if ord(prefix[-1]) > _MAX_INCREMENTABLE:
        return None
    return prefix[:-1] + chr(ord(prefix[-1]) + 1)


DEFAULT_REGISTRY.add(
    plan_rule(
        "like_prefix_to_range",
        Phase.NORMALIZE,
        lambda plan, _ctx: like_prefix_to_range(plan),
    )
)


def _flat_or_equalities(expr: Expr) -> tuple[str, list] | None:
    """If `expr` is `c == v1 OR c == v2 OR …` (≥2 disjuncts, one column, literal
    values), return `(column, [values])`; else None. The shape SQL `IN (...)` and
    chained `OR` equalities lower to."""
    if not (isinstance(expr, Binary) and expr.op == "or"):
        return None
    leaves: list[tuple[str, object]] = []

    def collect(e: Expr) -> bool:
        if isinstance(e, Binary) and e.op == "or":
            return collect(e.left) and collect(e.right)
        if isinstance(e, Binary) and e.op == "eq":
            left, right = e.left, e.right
            if isinstance(left, Col) and isinstance(right, Lit):
                leaves.append((left.name, right.value))
                return True
            if isinstance(right, Col) and isinstance(left, Lit):
                leaves.append((right.name, left.value))
                return True
        return False

    if not collect(expr) or len(leaves) < 2:
        return None
    cols = {name for name, _ in leaves}
    values = [v for _, v in leaves]
    if len(cols) != 1 or any(v is None or isinstance(v, bool) for v in values):
        return None
    return cols.pop(), values


@rule(name="or_to_in_and_range", phase=Phase.NORMALIZE, matches=(Filter,))
def or_to_in_and_range(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Add `c >= min AND c <= max` alongside a `c = v1 OR c = v2 OR …` conjunct.

    A disjunction of equalities (what `IN (...)` lowers to) is opaque to range-based
    zone-map pruning. Its values imply the bound `min(vs) ≤ c ≤ max(vs)`, a superset
    that — ANDed with the original disjunction — leaves the result unchanged but gives
    `zonemap_prune_filter` a range it can use to skip whole row groups (and each
    equality is still a bloom-index probe). Idempotent: the bounds are added only if
    not already present. Skipped when the literals aren't mutually comparable.
    """
    conjuncts = split_conjuncts(node.predicate)
    existing = [c.to_ir() for c in conjuncts]  # IR dicts are unhashable → list + `in`
    added: list[Expr] = []
    for conj in conjuncts:
        info = _flat_or_equalities(conj)
        if info is None:
            continue
        col_name, values = info
        try:
            lo, hi = min(values), max(values)
        except TypeError:
            continue  # values not mutually comparable (mixed types)
        for bound in (Binary("ge", Col(col_name), Lit(lo)), Binary("le", Col(col_name), Lit(hi))):
            if bound.to_ir() not in existing:
                added.append(bound)
                existing.append(bound.to_ir())
    if not added:
        return None
    return Filter(node.input, combine_conjuncts([*conjuncts, *added]))
