"""Arithmetic algebraic simplification — integer constant reassociation & factoring.

NORMALIZE-phase, node-local rules over the expressions a `Filter` or `Project`
carries. They collapse redundant arithmetic structure the SQL/DataFrame front end
(or a prior rewrite) leaves behind: nested `+`/`-`/`*` against constants fold into a
single operation, a value subtracted from a constant is canonicalized, a
subtraction-from-zero (i.e. negation) is peeled, and a common multiplicand is
factored out of a sum/difference of products.

Correctness is anchored to the engine's *exact* arithmetic semantics, which is why
every rule here is gated on a **signed-integer type guard**:

* `bc-expr` evaluates `add`/`sub`/`mul` with **two's-complement wrapping** i64
  arithmetic (`add_wrapping`/`sub_wrapping`/`mul_wrapping`, bit-identical with the
  Cranelift JIT — it never errors or promotes on overflow). The integers modulo
  ``2**64`` form a commutative ring, so `+`/`-`/`*` are associative, commutative and
  distributive there **even across overflow** — reassociating is exact, including at
  the ``INT64_MIN``/``INT64_MAX`` boundary. A folded constant is wrapped back into
  i64 so `mul_wrapping(x, wrap(c1*c2)) == (x*c1)*c2 (mod 2**64)`.
* Float arithmetic is IEEE-754 (rounding, NaN, ±inf, ±0), where none of these laws
  hold (`(x+1)-1 != x` at ``2**53``; `0-(a-b)` flips the sign of a zero). So a rule
  fires only when the whole arithmetic subtree is provably signed-integer-typed —
  every column an int per the input schema, every literal a non-bool int, every
  operator in `{add, sub, mul}`. Anything float / decimal / unknown-typed is left
  untouched.

NULL is preserved by construction: these are the same arithmetic operators over the
same operands (`x + c`, `x * c`, `c - x`, `b - a`), so a null input stays null — no
rule ever introduces a literal `0`/`1` in place of a possibly-null value.

These do **not** duplicate `normalize.ExprSimplification`, which already drops the
single-operator identities `x + 0`, `0 + x`, `x - 0`, `x * 1`, `1 * x`, `x / 1`
(and `NOT NOT`, `Cast(Cast)`); here every rule combines **two** operators (needing
no identity element) or factors, which that pass does not do.
"""

from __future__ import annotations

import pyarrow as pa

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.plan.expr_ir import Binary, Col, Expr, Lit
from batcher.plan.expr_rewrite import map_node_expressions, transform_expr_up
from batcher.plan.logical import Filter, LogicalPlan, Project
from batcher.plan.schema import SchemaRef

__all__ = [
    "factor_common_mul",
    "fold_add_sub_constants",
    "fold_const_minus_sum",
    "fold_mul_constants",
    "fold_neg_sub",
]

_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1


# --- shared helpers ---------------------------------------------------------


def _int_cols(schema: SchemaRef | None) -> frozenset[str]:
    """Names of the schema's **signed-integer** columns (int8/16/32/64 → i64 in the
    engine). An empty set — returned when the schema is unknown — makes every rule a
    no-op, which is the safe default."""
    if schema is None:
        return frozenset()
    return frozenset(f.name for f in schema.arrow if pa.types.is_signed_integer(f.type))


def _is_int_lit(expr: Expr) -> bool:
    """Whether `expr` is a non-bool integer literal (bool is an `int` subclass)."""
    return isinstance(expr, Lit) and type(expr.value) is int


def _is_int_arith(expr: Expr, int_cols: frozenset[str]) -> bool:
    """Whether `expr` is provably signed-integer-typed: only int columns, non-bool
    int literals, and `add`/`sub`/`mul` operators. Under this guard the engine
    evaluates the whole subtree with wrapping i64 arithmetic, so the ring laws
    (associativity/commutativity/distributivity mod 2**64) the rules rely on hold
    exactly."""
    if _is_int_lit(expr):
        return True
    if isinstance(expr, Col):
        return expr.name in int_cols
    if isinstance(expr, Binary) and expr.op in ("add", "sub", "mul"):
        return _is_int_arith(expr.left, int_cols) and _is_int_arith(expr.right, int_cols)
    return False


def _wrap_i64(k: int) -> int:
    """Reduce `k` into the signed 64-bit range (two's-complement wraparound), so a
    folded constant matches what wrapping i64 arithmetic would compute."""
    return ((k + 2**63) % 2**64) - 2**63


def _offset(v: Expr, k: int) -> Expr:
    """Build the canonical `v + k` (wrapping): `v` when `k == 0`, else `v + k` /
    `v - (-k)`. Keeps the additive constant on the right with a non-negative literal
    where possible; `INT64_MIN` stays an `add` because its negation overflows."""
    k = _wrap_i64(k)
    if k == 0:
        return v
    if k > 0 or k == _INT64_MIN:
        return Binary("add", v, Lit(k))
    return Binary("sub", v, Lit(-k))


def _var_offset(expr: Expr, int_cols: frozenset[str]) -> tuple[Expr, int] | None:
    """Match `v + c` / `c + v` / `v - c` (one non-bool int literal, the other operand
    a non-literal int-arith `v`) → `(v, signed_c)` with `expr == v + signed_c`. `None`
    otherwise (notably `c - v`, handled by `fold_const_minus_sum`)."""
    if not (isinstance(expr, Binary) and expr.op in ("add", "sub")):
        return None
    left, right = expr.left, expr.right
    if expr.op == "add":
        if _is_int_lit(left) and not _is_int_lit(right) and _is_int_arith(right, int_cols):
            return right, left.value
        if _is_int_lit(right) and not _is_int_lit(left) and _is_int_arith(left, int_cols):
            return left, right.value
    elif _is_int_lit(right) and not _is_int_lit(left) and _is_int_arith(left, int_cols):
        return left, -right.value
    return None


def _var_factor(expr: Expr, int_cols: frozenset[str]) -> tuple[Expr, int] | None:
    """Match `v * c` / `c * v` (one non-bool int literal, the other a non-literal
    int-arith `v`) → `(v, c)` with `expr == v * c`. `None` otherwise."""
    if not (isinstance(expr, Binary) and expr.op == "mul"):
        return None
    left, right = expr.left, expr.right
    if _is_int_lit(left) and not _is_int_lit(right) and _is_int_arith(right, int_cols):
        return right, left.value
    if _is_int_lit(right) and not _is_int_lit(left) and _is_int_arith(left, int_cols):
        return left, right.value
    return None


def _apply(node: LogicalPlan, leaf) -> LogicalPlan | None:
    """Run `leaf` bottom-up over every expression `node` carries (using the input
    schema for the integer guard), returning the rebuilt node — or `None` when
    nothing changed, so the driver's fixpoint terminates."""
    input_node = node.input
    # The int-column set is a pure function of the input's schema, but `_apply` runs
    # once per arith rule per node per fixpoint iteration — recomputing it (a scan of
    # every schema field) each time is waste. Memoize it on the (immutable) input node,
    # the same `__dict__` cache `to_ir`/`available_schema` use.
    int_cols = input_node.__dict__.get("_c_int_cols")
    if int_cols is None:
        int_cols = _int_cols(input_node.available_schema())
        input_node.__dict__["_c_int_cols"] = int_cols

    def rewrite(expr: Expr) -> Expr:
        return transform_expr_up(expr, lambda e: leaf(e, int_cols))

    rebuilt = map_node_expressions(node, rewrite)
    # `map_node_expressions` returns the *same object* when no expression changed, so the
    # `is` check settles the common no-op case in O(1); the IR comparison is the fallback
    # for a rule that rebuilt an equal-but-new expression (identical result either way).
    return None if rebuilt is node or rebuilt.to_ir() == node.to_ir() else rebuilt


# --- fold_add_sub_constants -------------------------------------------------


def _combine_add_sub(expr: Expr, int_cols: frozenset[str]) -> Expr:
    outer = _var_offset(expr, int_cols)
    if outer is None:
        return expr
    inner_expr, k_outer = outer
    inner = _var_offset(inner_expr, int_cols)
    if inner is None:
        return expr
    v, k_inner = inner
    return _offset(v, k_inner + k_outer)


@rule(name="fold_add_sub_constants", phase=Phase.NORMALIZE, matches=(Filter, Project))
def fold_add_sub_constants(node: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Fold two nested additive integer constants into one: `(x + c1) + c2 → x + (c1+c2)`,
    `(x - c1) - c2 → x - (c1+c2)`, `(x + c1) - c2 → x + (c1-c2)`, and the cancellation
    `(x + c) - c → x`.

    Two's-complement addition is associative (the integers mod ``2**64`` are a ring),
    so combining the constants is exact even when an intermediate `x + c1` overflows —
    the engine's `add_wrapping`/`sub_wrapping` wrap identically to the folded form. The
    combined constant is wrapped back into i64. Gated on a signed-integer type guard
    because float `+`/`-` is *not* associative (`(x+1)-1 != x` near ``2**53``). NULL is
    preserved (a null `x` stays null on both sides). Bottom-up traversal collapses a
    whole chain in one pass; nested products (`x*3`) ride through untouched as `x`.
    """
    return _apply(node, _combine_add_sub)


# --- fold_mul_constants -----------------------------------------------------


def _combine_mul(expr: Expr, int_cols: frozenset[str]) -> Expr:
    outer = _var_factor(expr, int_cols)
    if outer is None:
        return expr
    inner_expr, c_outer = outer
    inner = _var_factor(inner_expr, int_cols)
    if inner is None:
        return expr
    v, c_inner = inner
    prod = _wrap_i64(c_inner * c_outer)
    return v if prod == 1 else Binary("mul", v, Lit(prod))


@rule(name="fold_mul_constants", phase=Phase.NORMALIZE, matches=(Filter, Project))
def fold_mul_constants(node: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Fold two nested integer factors into one: `(x * c1) * c2 → x * (c1*c2)`, with the
    cancellation `(x * c1) * c2 → x` when `c1*c2 ≡ 1` (mod ``2**64``).

    Two's-complement multiplication is associative (ring mod ``2**64``), so the fold is
    exact even when `x * c1` overflows; the product constant is wrapped back into i64.
    Gated on the signed-integer guard because float `*` is not associative. NULL is
    preserved — including the `c1*c2 ≡ 0` case, which is emitted as `x * 0` (still null
    where `x` is null), never the literal `0`. `x * 1` is left to `ExprSimplification`
    only when it *arrives* as a lone identity; here the `≡ 1` product is reduced directly.
    """
    return _apply(node, _combine_mul)


# --- fold_const_minus_sum ---------------------------------------------------


def _combine_const_minus_sum(expr: Expr, int_cols: frozenset[str]) -> Expr:
    if not (isinstance(expr, Binary) and expr.op == "sub" and _is_int_lit(expr.left)):
        return expr
    inner = _var_offset(expr.right, int_cols)
    if inner is None:
        return expr
    v, k = inner
    # c1 - (v + k) == (c1 - k) - v  (exact under wrapping subtraction).
    return Binary("sub", Lit(_wrap_i64(expr.left.value - k)), v)


@rule(name="fold_const_minus_sum", phase=Phase.NORMALIZE, matches=(Filter, Project))
def fold_const_minus_sum(node: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Canonicalize a constant minus a value-plus-constant: `c1 - (x + c2) → (c1-c2) - x`
    and `c1 - (x - c2) → (c1+c2) - x`.

    `c1 - (x ± c2) == (c1 ∓ c2) - x` exactly under wrapping i64 subtraction (ring mod
    ``2**64``); the residual constant is wrapped back into i64. This hoists the two
    constants together (feeding further folding / constant propagation) and leaves the
    value in a single `const - x` shape. Gated on the signed-integer guard (float `-`
    reassociation is unsafe). NULL is preserved (`x` null → null on both sides).
    """
    return _apply(node, _combine_const_minus_sum)


# --- fold_neg_sub -----------------------------------------------------------


def _combine_neg_sub(expr: Expr, int_cols: frozenset[str]) -> Expr:
    if not (isinstance(expr, Binary) and expr.op == "sub"):
        return expr
    left, inner = expr.left, expr.right
    if not (_is_int_lit(left) and left.value == 0):
        return expr
    if not (isinstance(inner, Binary) and inner.op == "sub"):
        return expr
    a, b = inner.left, inner.right
    if _is_int_arith(a, int_cols) and _is_int_arith(b, int_cols):
        return Binary("sub", b, a)  # 0 - (a - b) == b - a
    return expr


@rule(name="fold_neg_sub", phase=Phase.NORMALIZE, matches=(Filter, Project))
def fold_neg_sub(node: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Peel a negation of a difference: `0 - (a - b) → b - a` (which also collapses the
    double negation `-(-x)`, since `-x` desugars to `0 - x`, so `-(-x) = 0 - (0 - x) → x - 0`).

    `0 - (a - b) == b - a` exactly under wrapping i64 subtraction — the engine negates a
    difference bit-for-bit into the swapped subtraction. Gated on the signed-integer
    guard: for floats `0 - (a - b)` and `b - a` disagree on the sign of a zero when
    `a == b`. NULL is preserved (either operand null → null on both sides). The leftover
    `x - 0` is finished by `ExprSimplification`.
    """
    return _apply(node, _combine_neg_sub)


# --- factor_common_mul ------------------------------------------------------


def _combine_factor_mul(expr: Expr, int_cols: frozenset[str]) -> Expr:
    if not (isinstance(expr, Binary) and expr.op in ("add", "sub")):
        return expr
    left = _var_factor(expr.left, int_cols)
    right = _var_factor(expr.right, int_cols)
    if left is None or right is None:
        return expr
    v1, c1 = left
    v2, c2 = right
    if v1.to_ir() != v2.to_ir():
        return expr
    coeff = _wrap_i64(c1 + c2 if expr.op == "add" else c1 - c2)
    return v1 if coeff == 1 else Binary("mul", v1, Lit(coeff))


@rule(name="factor_common_mul", phase=Phase.NORMALIZE, matches=(Filter, Project))
def factor_common_mul(node: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Factor a common integer multiplicand out of a sum/difference of products:
    `x*c1 + x*c2 → x*(c1+c2)` and `x*c1 - x*c2 → x*(c1-c2)` (with `x` structurally equal
    on both sides), reducing to `x` when the resulting coefficient is `1`.

    Distributivity holds exactly in the ring mod ``2**64``, so the factoring is exact
    even across overflow; the coefficient is wrapped back into i64. Gated on the
    signed-integer guard (float `*`/`+` do not distribute exactly). NULL is preserved —
    a null `x` makes both the two-product form and `x*coeff` null (the `coeff ≡ 0` case
    is emitted as `x * 0`, never the literal `0`). One evaluation of `x` instead of two.
    """
    return _apply(node, _combine_factor_mul)
