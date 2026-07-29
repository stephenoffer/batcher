"""Constant folding of string functions over string literals.

Split out of `extra/strings`, which holds the *pattern* rewrites (`LIKE` to an affix test,
idempotent-call collapsing, trim absorption). These four are the other half of that family:
a string function whose whole argument is already a literal is evaluated at plan time, so
the engine never runs the kernel per row.

Registration order is run order, so this module is imported from `extra/__init__` directly
after `strings` -- exactly the position these rules occupied when the two were one file.
"""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.kyber.rules.extra.strings import _apply, _str_lit
from batcher.plan.expr_ir import Binary, Expr, Lit, StrFunc
from batcher.plan.logical import Filter, LogicalPlan, Project

__all__ = [
    "fold_case_of_literal",
    "fold_concat_of_literals",
    "fold_len_of_literal",
    "fold_substr_of_literal",
]


def _leaf_fold_case(expr: Expr, _str_cols: frozenset[str]) -> Expr:
    if not (isinstance(expr, StrFunc) and expr.fn in ("upper", "lower")):
        return expr
    value = _str_lit(expr.input)
    # ASCII only: Python's and Rust's *full* Unicode case mappings are not guaranteed to agree,
    # and the engine — not Python — defines the result.
    if value is None or not value.isascii():
        return expr
    return Lit(value.upper() if expr.fn == "upper" else value.lower())


@rule(
    name="fold_case_of_literal",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr_matches=(StrFunc,),
    expr_ops=("lower", "upper"),
)
def fold_case_of_literal(node: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Fold `upper('abc')` → `'ABC'` and `lower('ABC')` → `'abc'` at plan time.

    A case conversion of a constant is a constant. Restricted to an **ASCII** literal, where
    Python's `str.upper`/`str.lower` and the engine's `to_uppercase`/`to_lowercase` are provably
    the same function (a 1:1 map over `[A-Za-z]`); non-ASCII is left to the engine, since full
    Unicode case mapping is where implementations diverge. The literal is non-null; type unchanged.
    """
    return _apply(node, _leaf_fold_case)


def _leaf_fold_len(expr: Expr, _str_cols: frozenset[str]) -> Expr:
    if not (isinstance(expr, StrFunc) and expr.fn in ("len", "octet_length", "bit_length")):
        return expr
    value = _str_lit(expr.input)
    if value is None:
        return expr
    if expr.fn == "len":
        return Lit(len(value))
    octets = len(value.encode("utf-8"))
    return Lit(octets if expr.fn == "octet_length" else octets * 8)


@rule(
    name="fold_len_of_literal",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr_matches=(StrFunc,),
    expr_ops=("bit_length", "len", "octet_length"),
)
def fold_len_of_literal(node: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Fold a string literal's length: `len('héllo')` → `5`, `octet_length('héllo')` → `6`.

    The engine's `len` counts Unicode scalar values (`chars().count()`, matching DuckDB's `length`)
    and Python's `len(s)` counts code points — the same number for every string that can cross the
    UTF-8 FFI boundary. `octet_length` is the UTF-8 byte count and `bit_length` eight times that,
    exactly as the engine computes them (`bit_length` too — eight times the byte count). The
    literal is non-null; the fold keeps the Int64 type.
    """
    return _apply(node, _leaf_fold_len)


def _leaf_fold_concat(expr: Expr, _str_cols: frozenset[str]) -> Expr:
    if not (isinstance(expr, Binary) and expr.op == "concat"):
        return expr
    left, right = _str_lit(expr.left), _str_lit(expr.right)
    return expr if left is None or right is None else Lit(left + right)


@rule(
    name="fold_concat_of_literals",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr_matches=(Binary,),
    expr_ops=("concat",),
)
def fold_concat_of_literals(node: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Fold a concatenation of string literals: `'a' || 'b'` → `'ab'`.

    Arrow's `concat_elements_utf8` — what the engine's `concat` (SQL `||`) runs — is plain byte
    concatenation on Utf8, which `+` on two Python `str`s reproduces exactly. Fires only when *both*
    sides are string literals: the engine *casts* a non-Utf8 operand to Utf8 first and Arrow's
    number→string formatting is not Python's. `ConstantFolding` declines `concat`, so nothing else
    folds this; a NULL operand makes `||` null, and `Lit(None)` is not matched here.
    """
    return _apply(node, _leaf_fold_concat)


def _substr(value: str, start: int, length: int | None) -> str:
    """`substr` with the engine's (DuckDB's) semantics, mirroring `bc_expr`'s `substr_slice` step
    for step: 1-based; a negative `start` counts from the end (`n + start + 1`); a positive `length`
    spans the inclusive window `[start, start + length - 1]`, a negative one flips it to
    `[start + length, start - 1]`; the window is *clipped* to `[1, n]` (out-of-range positions are
    dropped, not shifted) and an empty intersection yields `''`. Indexed by code point — which
    Python does natively, exactly as the engine's `chars()` does."""
    n = len(value)
    begin = n + start + 1 if start < 0 else start
    if length is None:
        lo, hi = begin, n
    elif length >= 0:
        lo, hi = begin, begin + length - 1
    else:
        lo, hi = begin + length, begin - 1
    lo, hi = max(lo, 1), min(hi, n)
    return "" if hi < lo else value[lo - 1 : hi]


def _leaf_fold_substr(expr: Expr, _str_cols: frozenset[str]) -> Expr:
    if not (isinstance(expr, StrFunc) and expr.fn == "substr"):
        return expr
    value = _str_lit(expr.input)
    if value is None:
        return expr
    # `start` defaults to 1 in the engine when the IR omits it.
    return Lit(_substr(value, expr.start if expr.start is not None else 1, expr.length))


@rule(
    name="fold_substr_of_literal",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr_matches=(StrFunc,),
    expr_ops=("substr",),
)
def fold_substr_of_literal(node: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Fold a substring of a string literal: `substr('abcdef', 2, 3)` → `'bcd'`.

    `_substr` reimplements the engine's `substr_slice` exactly (see its docstring), and the edge
    cases are where that matters: `substr('abcdef', 0, 3)` is `'ab'` — the window `[0, 2]` clips to
    `[1, 2]` rather than shifting — and `substr('abcdef', -2, 4)` is `'ef'`. The literal is
    non-null; the fold keeps the Utf8 type.
    """
    return _apply(node, _leaf_fold_substr)
