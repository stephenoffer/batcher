"""Equi-join keys over *expressions*: `a.x = b.y - 52` as a hash key, not a post-join filter.

`derive_join_keys` turns `left_col = right_col` conjuncts above an inner join into join keys,
and only those: a key is a column *name*, so an equality with an expression on either side
stayed in the filter and was evaluated after the join had been built on whatever keys were
left. When the remaining keys are coarse that is a near-cartesian intermediate. TPC-DS q59
joins two week-by-store rollups on `s_store_id1 = s_store_id2 AND d_week_seq1 = d_week_seq2 -
52`: on the store key alone the join emits 799,350 rows, and the week filter above it keeps
15,288. DuckDB joins on both.

The fix gives the expression a name. Each side that needs one gains a `Project` that carries
every column through and computes the operand as a hidden key column, and the join keys on
it. The join's output list is unchanged, so the hidden column never escapes the join.

**Which expressions qualify is deliberately narrow.** Moving an expression below the join
evaluates it once per *input* row instead of once per *joined* row. For that to be
invisible it must be deterministic and must not be able to raise on a row the join would
have discarded. An operand qualifies only if it is a column, or a column plus or minus an
integer literal of modest magnitude, which is the shape the week- and year-offset joins of
TPC-DS take. Division, multiplication, casts and every function are refused. The key's type
must also be one whose hash-join equality is exactly SQL `=`, so both operands must infer the
same integer, date or string type. Floats are excluded, because a hash key canonicalizes NaN
and -0.0 where `=` does not. NULL needs no special case: an inner join never matches a NULL
key, and a NULL comparison fails the filter the same way.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pyarrow as pa

from batcher.plan.expr_ir import Binary, Col, Expr, Lit, referenced_columns, remap_columns
from batcher.plan.logical import Join, LogicalPlan, Project, Projection
from batcher.plan.types import infer_type

__all__ = ["ExprKeyPair", "attach_expr_keys", "expr_key_pair"]

# Beyond this an `x + k` offset is no longer the calendar arithmetic the rule exists for, and
# it starts to approach the range where an `Int64` add could overflow on a row the join
# would have discarded. 2**31 keeps every qualifying add exact for any 32-bit-origin key.
_MAX_OFFSET = 1 << 31

# Prefix for the hidden key columns. Double-underscore names are the engine's reserved space
# (`__cross_key`, `__gk0`), and the digest makes a collision with a user column impossible.
_KEY_PREFIX = "__eqk_"


@dataclass(frozen=True, eq=False)
class ExprKeyPair:
    """One cross-side equality, each operand phrased in its own side's *source* names.

    Compared by identity (`eq=False`): its fields are `Expr`s, whose `==` builds an
    expression instead of comparing, so a generated `__eq__` would call every pair equal.
    """

    left: Expr
    right: Expr


def _qualifies(expr: Expr) -> bool:
    """Whether `expr` is a column, or a column plus/minus a small integer literal."""
    if isinstance(expr, Col):
        return True
    if not (isinstance(expr, Binary) and expr.op in ("add", "sub")):
        return False
    col, lit = expr.left, expr.right
    if expr.op == "add" and isinstance(col, Lit):
        col, lit = lit, col
    return (
        isinstance(col, Col)
        and isinstance(lit, Lit)
        and isinstance(lit.value, int)
        and not isinstance(lit.value, bool)
        and abs(lit.value) < _MAX_OFFSET
    )


def expr_key_pair(
    conj: Expr, left_src: dict[str, str], right_src: dict[str, str]
) -> ExprKeyPair | None:
    """The equality `conj` as a cross-side key pair with at least one expression operand.

    `left_src`/`right_src` map the join's output aliases to each side's source names, as in
    `derive_join_keys`. A `col = col` conjunct is that rule's own case and returns `None`
    here, as does anything outside the shapes `_qualifies` admits.

    Args:
        conj: One conjunct of the filter above the join.
        left_src: Left-side output alias -> source column name.
        right_src: Right-side output alias -> source column name.

    Returns:
        The pair, rewritten into source names, or `None`.
    """
    if not isinstance(conj, Binary) or conj.op != "eq":
        return None
    lhs, rhs = conj.left, conj.right
    if isinstance(lhs, Col) and isinstance(rhs, Col):
        return None
    if not (_qualifies(lhs) and _qualifies(rhs)):
        return None
    lrefs, rrefs = referenced_columns(lhs), referenced_columns(rhs)
    if lrefs <= left_src.keys() and rrefs <= right_src.keys():
        pair = (lhs, rhs)
    elif lrefs <= right_src.keys() and rrefs <= left_src.keys():
        pair = (rhs, lhs)
    else:
        return None
    return ExprKeyPair(remap_columns(pair[0], left_src), remap_columns(pair[1], right_src))


def _key_type_ok(left: pa.DataType | None, right: pa.DataType | None) -> bool:
    """Both operands infer one type whose hash-key equality is exactly SQL `=`."""
    if left is None or right is None or left != right:
        return False
    return (
        pa.types.is_integer(left)
        or pa.types.is_date(left)
        or pa.types.is_string(left)
        or pa.types.is_large_string(left)
    )


def _key_name(expr: Expr) -> str:
    return _KEY_PREFIX + hashlib.sha1(repr(expr.to_ir()).encode()).hexdigest()[:12]


def _with_keys(side: LogicalPlan, exprs: list[Expr]) -> tuple[LogicalPlan, list[str]]:
    """`side` carrying a hidden column for each non-column expression, and the key names."""
    names: list[str] = []
    extra: dict[str, Expr] = {}
    for e in exprs:
        if isinstance(e, Col):
            names.append(e.name)
            continue
        name = _key_name(e)
        names.append(name)
        extra.setdefault(name, e)
    if not extra:
        return side, names
    items = tuple(Projection(c, Col(c)) for c in side.available_columns())
    items += tuple(Projection(n, e) for n, e in extra.items())
    return Project(side, items), names


def attach_expr_keys(join: Join, pairs: list[ExprKeyPair]) -> tuple[Join, list[ExprKeyPair]]:
    """`join` keyed additionally on every pair whose types allow it, and the pairs refused.

    The refused pairs' conjuncts must stay in the filter above the join; the caller keeps
    them, so a refusal costs only the rewrite, never the predicate.

    Args:
        join: An inner join.
        pairs: Candidate pairs from `expr_key_pair`.

    Returns:
        The rewritten join (the input join when nothing was accepted) and the refused pairs.
    """
    lschema, rschema = join.left.available_schema(), join.right.available_schema()
    if lschema is None or rschema is None:
        return join, pairs
    taken: list[ExprKeyPair] = []
    refused: list[ExprKeyPair] = []
    for p in pairs:
        ok = _key_type_ok(infer_type(p.left, lschema), infer_type(p.right, rschema))
        (taken if ok else refused).append(p)
    if not taken:
        return join, pairs
    left, lnames = _with_keys(join.left, [p.left for p in taken])
    right, rnames = _with_keys(join.right, [p.right for p in taken])
    keyed = Join(
        left,
        right,
        join.left_keys + tuple(lnames),
        join.right_keys + tuple(rnames),
        join.join_type,
        join.output,
        join.strategy,
    )
    return keyed, refused
