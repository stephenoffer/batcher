"""NORMALIZE-phase sargable-predicate normalization — strip arithmetic wrappers so a
predicate becomes a bare `col OP literal` that zone-map pruning and source pushdown can use.

A predicate like `col + 100 = 500` or `col * 3 = 9` is opaque to data skipping: the
zone-map / bloom machinery (`zonemap_prune_filter`) and source predicate pushdown only
recognize a comparison whose column side is a *raw* `Col`. These rewrites peel the
constant arithmetic off the column and fold it into the literal, exposing the raw column
so the downstream sargability passes can prune whole row groups / files.

Correctness is anchored to the engine's own arithmetic, which is **wrapping** on i64
overflow (`bc_expr` uses `add_wrapping`/`sub_wrapping`/`mul_wrapping`, bit-for-bit with the
Cranelift JIT's `iadd`/`isub`/`imul`; it does *not* error or promote to a wider type). That
single fact bounds what is provably exact:

* **Equality / inequality only for the additive and multiplicative forms.** `add`/`sub`
  by a constant is a bijection of `Z/2^64`, so `col + k = lit  <=>  col = lit - k` holds for
  *every* column value (the wrap cancels) — but only for `=`/`<>`. The *ordered* forms
  (`col + k < lit -> col < lit - k`) are **not** rewritten: wrapping breaks monotonicity
  (`col = INT64_MAX`, `col + 5 > 10` is false while `col > 5` is true), so that transform
  would change results. Multiplication is a bijection only for an **odd** coefficient, and
  the unique pre-image is `lit / k` only when `k` exactly divides `lit`; an even `k`
  (non-injective mod 2^64) and a non-divisible literal are left untouched.
* **Integer literals only.** Floats are never rewritten (the engine's rounding would differ
  from Python's folded constant); dates/timestamps carry no `+/- int` here.
* **The folded literal is overflow-guarded.** A rewrite fires only when the new literal
  (`lit - k`, `lit + k`, `k - lit`) stays within i64, so it never introduces a value the
  engine would itself wrap.

The always-exact comparison flip (`lit OP col -> col flip(OP) lit`) carries no arithmetic
and is applied unconditionally, canonicalizing the literal to the right so the raw-column
passes and the rules above see a uniform shape.

Each rewrite is a `plan_rule` in `Phase.NORMALIZE` (registered into `DEFAULT_REGISTRY`
below), applied to every expression in the tree via `map_node_expressions` +
`transform_expr_up`. Their pure functions stay importable for unit tests.
"""

from __future__ import annotations

from collections.abc import Callable

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rule import Phase, node_rule
from batcher.plan.expr_ir import Binary, Col, Expr, Lit
from batcher.plan.expr_rewrite import map_node_expressions, transform_expr_up
from batcher.plan.logical import Aggregate, Filter, LogicalPlan, Project, Sort, Window
from batcher.plan.visitor import transform_up

__all__ = [
    "flip_comparison_literal",
    "sarg_add_const",
    "sarg_mul_const",
    "sarg_rsub_const",
    "sarg_sub_const",
    "sarg_xor_const",
]

_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1
# Comparisons that flip when the column moves from the right side to the left.
_FLIP = {"eq": "eq", "ne": "ne", "lt": "gt", "gt": "lt", "le": "ge", "ge": "le"}
# The additive/multiplicative strength reductions are exact *only* for these ops (a wrap of
# the arithmetic would break an ordered comparison; equality's bijection is wrap-invariant).
_EQ_NE = frozenset({"eq", "ne"})

ExprRule = Callable[[Expr], Expr]


def _is_int(value: object) -> bool:
    """Whether `value` is a plain Python int (bool — an int subclass — excluded)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _in_int64(value: int) -> bool:
    """Whether a folded literal is representable as i64 (so the engine won't itself wrap it)."""
    return _INT64_MIN <= value <= _INT64_MAX


def _expr_pass(leaf: ExprRule) -> Callable[[LogicalPlan], LogicalPlan]:
    """A whole-plan pass applying `leaf` bottom-up to every sub-expression of every node."""

    def run(plan: LogicalPlan) -> LogicalPlan:
        return transform_up(
            plan, lambda node: map_node_expressions(node, lambda e: transform_expr_up(e, leaf))
        )

    return run


def _arith_and_lit(expr: Binary) -> tuple[Binary, int] | None:
    """`(inner_binary, literal)` for a `= / <>` whose one side is a `Binary` and the other an
    int `Lit` (either order — equality is symmetric), else None."""
    left, right = expr.left, expr.right
    if isinstance(right, Lit) and isinstance(left, Binary) and _is_int(right.value):
        return left, right.value
    if isinstance(left, Lit) and isinstance(right, Binary) and _is_int(left.value):
        return right, left.value
    return None


def _commutative_col_const(inner: Binary) -> tuple[Col, int] | None:
    """`(col, k)` for `col * k` or `k * col` (a commutative op: `add`/`mul`/`bit_xor`)."""
    left, right = inner.left, inner.right
    if isinstance(left, Col) and isinstance(right, Lit) and _is_int(right.value):
        return left, right.value
    if isinstance(right, Col) and isinstance(left, Lit) and _is_int(left.value):
        return right, left.value
    return None


# --- comparison flip (always exact, no arithmetic) --------------------------


def flip_comparison_literal(plan: LogicalPlan) -> LogicalPlan:
    """Rewrite `literal OP col` to the canonical `col flip(OP) literal`.

    Flipping the operands of a comparison never changes its (three-valued) result, so this
    is unconditionally exact. It canonicalizes the literal to the right-hand side — the
    shape `comparison_col_side`, source predicate pushdown, and the additive rules below all
    expect — so a predicate a user (or the SQL front end) wrote as `500 = col` still reaches
    zone-map pruning. Fires only for a bare `literal OP col`; a `col`-on-left comparison is
    already canonical (so the rule is idempotent).
    """
    return _expr_pass(_flip_leaf)(plan)


def _flip_leaf(expr: Expr) -> Expr:
    if (
        isinstance(expr, Binary)
        and expr.op in _FLIP
        and isinstance(expr.left, Lit)
        and isinstance(expr.right, Col)
    ):
        return Binary(_FLIP[expr.op], expr.right, expr.left)
    return expr


# --- additive strength reduction (eq/ne only, overflow-guarded) -------------


def sarg_add_const(plan: LogicalPlan) -> LogicalPlan:
    """Rewrite `col + k = lit` (or `k + col = lit`) to `col = lit - k`, and the `<>` form.

    Addition by a constant is a bijection of `Z/2^64`, so the equality holds for the same
    rows before and after — the engine's wrapping `+` cancels exactly. Restricted to
    `=`/`<>` (an ordered comparison is not wrap-invariant) and to integer operands, and it
    fires only when `lit - k` stays within i64 (else the folded literal would itself wrap).
    Exposes the raw `col` for zone-map / bloom pruning.
    """
    return _expr_pass(_add_leaf)(plan)


def _add_leaf(expr: Expr) -> Expr:
    reduced = _reduce_additive(expr, "add", lambda lit, k: lit - k)
    return reduced if reduced is not None else expr


def sarg_sub_const(plan: LogicalPlan) -> LogicalPlan:
    """Rewrite `col - k = lit` to `col = lit + k`, and the `<>` form (eq/ne only, i64-guarded).

    The subtractive twin of `sarg_add_const`: `col - k` is a bijection, so the equality is
    preserved for every column value under the engine's wrapping subtraction. Fires only
    when `lit + k` fits in i64.
    """
    return _expr_pass(_sub_leaf)(plan)


def _sub_leaf(expr: Expr) -> Expr:
    match = _arith_and_lit(expr) if isinstance(expr, Binary) and expr.op in _EQ_NE else None
    if match is None:
        return expr
    inner, lit = match
    if inner.op != "sub" or not (isinstance(inner.left, Col) and isinstance(inner.right, Lit)):
        return expr
    if not _is_int(inner.right.value):
        return expr
    folded = lit + inner.right.value
    if not _in_int64(folded):
        return expr
    return Binary(expr.op, inner.left, Lit(folded))


def sarg_rsub_const(plan: LogicalPlan) -> LogicalPlan:
    """Rewrite `k - col = lit` to `col = k - lit`, and the `<>` form (eq/ne only, i64-guarded).

    Covers reverse subtraction and, as the `k = 0` case, unary minus (`-col`, which lowers to
    `0 - col`). `col -> k - col` is a bijection of `Z/2^64`, so the equality is preserved for
    every column value under wrapping subtraction. Fires only when `k - lit` fits in i64 (the
    guard that, for unary minus, declines to rewrite `-col = INT64_MIN`).
    """
    return _expr_pass(_rsub_leaf)(plan)


def _rsub_leaf(expr: Expr) -> Expr:
    match = _arith_and_lit(expr) if isinstance(expr, Binary) and expr.op in _EQ_NE else None
    if match is None:
        return expr
    inner, lit = match
    if inner.op != "sub" or not (isinstance(inner.right, Col) and isinstance(inner.left, Lit)):
        return expr
    if not _is_int(inner.left.value):
        return expr
    folded = inner.left.value - lit
    if not _in_int64(folded):
        return expr
    return Binary(expr.op, inner.right, Lit(folded))


def _reduce_additive(expr: Expr, op: str, fold: Callable[[int, int], int]) -> Binary | None:
    """Shared body for the commutative additive reduction (`add`): match `col * k <cmp> lit`
    and return `col <cmp> fold(lit, k)`, or None to leave it unchanged."""
    if not (isinstance(expr, Binary) and expr.op in _EQ_NE):
        return None
    match = _arith_and_lit(expr)
    if match is None:
        return None
    inner, lit = match
    if inner.op != op:
        return None
    col_const = _commutative_col_const(inner)
    if col_const is None:
        return None
    col, k = col_const
    folded = fold(lit, k)
    if not _in_int64(folded):
        return None
    return Binary(expr.op, col, Lit(folded))


# --- multiplicative strength reduction (eq/ne, odd coefficient, exact divide) ---


def sarg_mul_const(plan: LogicalPlan) -> LogicalPlan:
    """Rewrite `col * k = lit` to `col = lit / k`, and the `<>` form — only when it is exact.

    Multiplication is a bijection of `Z/2^64` iff `k` is **odd**; for an odd `k`, `col * k = lit`
    has the unique solution `col = lit / k` exactly when `k` divides `lit`. Both conditions are
    required: an even `k` (non-injective under wrapping — `col * 2 = 4` also matches
    `col = 2 + 2^63`) and a non-divisible literal are left untouched (rewriting either would
    change results). The quotient always fits in i64 (`|lit / k| <= |lit|`). The sign of `k`
    needs no operator flip here because the rule is `=`/`<>` only.
    """
    return _expr_pass(_mul_leaf)(plan)


def _mul_leaf(expr: Expr) -> Expr:
    if not (isinstance(expr, Binary) and expr.op in _EQ_NE):
        return expr
    match = _arith_and_lit(expr)
    if match is None:
        return expr
    inner, lit = match
    if inner.op != "mul":
        return expr
    col_const = _commutative_col_const(inner)
    if col_const is None:
        return expr
    col, k = col_const
    # Odd (⇒ bijection mod 2^64) and an exact divisor of `lit` (⇒ `lit / k` is the unique
    # pre-image). `k == 0` is even, so it is already excluded.
    if k % 2 == 0 or lit % k != 0:
        return expr
    return Binary(expr.op, col, Lit(lit // k))


# --- xor strength reduction (eq/ne; xor is its own inverse) ------------------


def sarg_xor_const(plan: LogicalPlan) -> LogicalPlan:
    """Rewrite `col ^ k = lit` (or `k ^ col = lit`) to `col = lit ^ k`, and the `<>` form.

    Bitwise xor by a constant is an involution — its own inverse — so `col ^ k = lit <=>
    col = lit ^ k` bit-for-bit, with no overflow possible (the result is a 64-bit pattern).
    Restricted to `=`/`<>` (xor is not order-preserving) and integer operands. Exposes the
    raw `col` for bloom / point-lookup skipping.
    """
    return _expr_pass(_xor_leaf)(plan)


def _xor_leaf(expr: Expr) -> Expr:
    reduced = _reduce_bitxor(expr)
    return reduced if reduced is not None else expr


def _reduce_bitxor(expr: Expr) -> Binary | None:
    if not (isinstance(expr, Binary) and expr.op in _EQ_NE):
        return None
    match = _arith_and_lit(expr)
    if match is None:
        return None
    inner, lit = match
    if inner.op != "bit_xor":
        return None
    col_const = _commutative_col_const(inner)
    if col_const is None:
        return None
    col, k = col_const
    folded = lit ^ k
    if not _in_int64(folded):
        return None
    return Binary(expr.op, col, Lit(folded))


# --- registration -----------------------------------------------------------

# Each of these walks the whole plan itself, and as a `plan_rule` it has no node type to be
# indexed on -- so without an expression declaration it runs against every plan there is.
# Every leaf gates on a `Binary` first: the five strength reductions need an `=`/`<>` (the
# only comparisons whose bijection survives the arithmetic), and the flip needs any
# comparison with the literal on the left.
#: The node types that carry expressions. `map_node_expressions` -- which `_expr_pass` and
#: the driver's fused chain both go through -- rewrites expressions on exactly these, and
#: returns `Scan`, `Join`, `Distinct`, `Union`, `Limit`, and `MapBatches` untouched. So
#: naming them here is not a narrowing: it is the same set the whole-plan pass already
#: reached, stated explicitly.
_EXPR_NODES = (Filter, Project, Aggregate, Sort, Window)


def _node_pass(leaf: ExprRule):
    """The node-local `f(node, ctx)` form of one leaf, for the driver's fused traversal."""

    def apply(node: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan | None:
        rebuilt = map_node_expressions(node, lambda e: transform_expr_up(e, leaf))
        return None if rebuilt is node else rebuilt

    return apply


# Registered as node rules rather than whole-plan ones so the driver runs all six inside the
# single expression traversal it already makes per node, instead of each walking the entire
# plan itself. The whole-plan `flip_comparison_literal`/`sarg_*_const` functions above stay
# as they are: they are the standalone form the unit tests drive directly.
for _name, _leaf, _ops in (
    ("sarg_flip_comparison", _flip_leaf, tuple(sorted(_FLIP))),
    ("sarg_add_const", _add_leaf, tuple(sorted(_EQ_NE))),
    ("sarg_sub_const", _sub_leaf, tuple(sorted(_EQ_NE))),
    ("sarg_rsub_const", _rsub_leaf, tuple(sorted(_EQ_NE))),
    ("sarg_mul_const", _mul_leaf, tuple(sorted(_EQ_NE))),
    ("sarg_xor_const", _xor_leaf, tuple(sorted(_EQ_NE))),
):
    DEFAULT_REGISTRY.add(
        node_rule(
            _name,
            Phase.NORMALIZE,
            _node_pass(_leaf),
            matches=_EXPR_NODES,
            expr_fn=_leaf,
            expr_matches=(Binary,),
            expr_ops=_ops,
        )
    )
