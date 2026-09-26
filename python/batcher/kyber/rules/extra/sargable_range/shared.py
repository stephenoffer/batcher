"""The decomposition and the overflow proof shared by the ordered sargable rules.

`sargable.py` transposes constant arithmetic across a comparison for `=`/`<>` only, and
its docstring says exactly why the *ordered* comparisons are excluded: the engine's i64
arithmetic **wraps**, so `col + 5 > 10` is false at `col = INT64_MAX` while `col > 5` is
true. Wrapping breaks monotonicity, and an ordered comparison needs monotonicity.

It breaks it only at the ends of the range, though. If every value the column can hold
keeps `col + k` inside i64, the addition is exact rather than modular, and the ordinary
integer identity `v + k OP lit  <=>  v OP lit - k` holds for every row. So the transform
is sound exactly when the plan can *prove* the arithmetic cannot wrap, and this module is
that proof. The range comes from the column's **measured min/max** — the bounds a Parquet
footer, ORC index, or lakehouse manifest records, which `RelStats` already carries and
zone-map pruning already trusts to delete whole row groups (`bounds.py`).

That range feeds a three-part obligation, checked per rewrite and never assumed:

1. the column side is a bare `Col` of provably integer type (the raw column is the point —
   it is what zone-map pruning and source pushdown recognize);
2. the arithmetic is exact over the whole range, i.e. both endpoints stay in i64;
3. the folded literal is itself representable in i64, so the rewrite introduces no value
   the engine would wrap.

Nulls need no separate argument: both sides are built from null-strict arithmetic and one
comparison over the same operand, so a null column yields a null answer either way.
"""

from __future__ import annotations

from batcher.plan.expr_ir import Binary, Col, Expr, Lit
from batcher.plan.expr_ir.core import int_literal
from batcher.plan.ir_tags import ORDERING_FLIP

__all__ = ["decompose", "transpose"]

_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1


def _split_inner(inner: Binary) -> tuple[str, Col, int] | None:
    """`(form, col, k)` for `col + k`, `k + col`, `col - k`, or `k - col`.

    `form` names which of the three the arithmetic is, because each one folds differently
    and each one has its own non-wrap obligation. The column side must be a bare `Col`:
    that is both what makes the range knowable and what the rewrite exists to expose.
    """
    left, right = inner.left, inner.right
    if inner.op == "add":  # commutative, so the literal may sit on either side
        for col, other in ((left, right), (right, left)):
            k = int_literal(other)
            if isinstance(col, Col) and k is not None:
                return "add", col, k
        return None
    if inner.op == "sub":
        k = int_literal(right)
        if isinstance(left, Col) and k is not None:
            return "sub", left, k
        k = int_literal(left)
        if isinstance(right, Col) and k is not None:
            return "rsub", right, k
    return None


def decompose(expr: Expr) -> tuple[str, str, Col, int, int] | None:
    """`(form, op, col, k, lit)` for an ordered comparison between `col ± k` and a literal.

    Normalizes the predicate so the arithmetic is conceptually on the left: a predicate
    written `lit > col + k` comes back with `op` flipped to `lt`, so a caller built for one
    operator sees both spellings without a second table.

    Args:
        expr: The candidate expression.

    Returns:
        The decomposition, or ``None`` when `expr` is not an ordered comparison between
        integer-constant arithmetic over a bare column and an integer literal.
    """
    if not isinstance(expr, Binary) or expr.op not in ORDERING_FLIP:
        return None
    for inner, other, op in (
        (expr.left, expr.right, expr.op),
        (expr.right, expr.left, ORDERING_FLIP[expr.op]),
    ):
        lit = int_literal(other)
        if lit is None or not isinstance(inner, Binary):
            continue
        split = _split_inner(inner)
        if split is not None:
            form, col, k = split
            return form, op, col, k, lit
    return None


def _in_int64(*values: int) -> bool:
    """Whether every value is representable as i64."""
    return all(_INT64_MIN <= v <= _INT64_MAX for v in values)


def transpose(form: str, op: str, col: Col, k: int, lit: int, low: int, high: int) -> Expr | None:
    """The transposed `col OP literal`, or ``None`` when the range does not prove it exact.

    This is obligations 2 and 3 from the module docstring, one branch per arithmetic form:

    * `col + k OP lit` -> `col OP lit - k`, exact while `low + k` and `high + k` are i64;
    * `col - k OP lit` -> `col OP lit + k`, exact while `low - k` and `high - k` are i64;
    * `k - col OP lit` -> `col ORDERING_FLIP(OP) k - lit`, exact while `k - low` and `k - high` are
      i64 — and with the operator flipped, because the column's coefficient is negative.

    Args:
        form: Which arithmetic shape `decompose` found (``add``, ``sub``, or ``rsub``).
        op: The comparison, already normalized to put the arithmetic on the left.
        col: The bare column the comparison should end up against.
        k: The integer constant.
        lit: The integer the comparison is against.
        low: The lowest value the column can hold.
        high: The highest value the column can hold.

    Returns:
        The rewritten comparison, or ``None`` when it would not be exact.
    """
    if form == "add":
        if not _in_int64(low + k, high + k, lit - k):
            return None
        return Binary(op, col, Lit(lit - k))
    if form == "sub":
        if not _in_int64(low - k, high - k, lit + k):
            return None
        return Binary(op, col, Lit(lit + k))
    if form == "rsub":
        if not _in_int64(k - low, k - high, k - lit):
            return None
        return Binary(ORDERING_FLIP[op], col, Lit(k - lit))
    return None
