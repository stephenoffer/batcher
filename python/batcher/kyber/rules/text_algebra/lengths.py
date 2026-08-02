"""A comparison against a string's length is almost always an emptiness test.

`length(s) > 0`, `octet_length(s) >= 1` and `bit_length(s) <> 0` all ask the same
question — is this string empty? — and none of them ask it in a form anything downstream
can use. `s <> ''` is a comparison against a literal, so `zonemap_prune_filter` can refute
a row group whose min and max are both the empty string, and a source can push the
predicate into the scan. The length call cannot be pushed anywhere.

All three functions are zero exactly for the empty string and positive for every other
non-null string, so the rewrite carries no information loss. It also carries no null
hazard: each is null-strict, and so is the comparison it becomes, which is why `>= 0` is
*absent* from the table. That one is `true` for every non-null string and `NULL` for a
null one, so its only faithful restatement is `s IS NOT NULL` — and that answers `false`
where the original answers `NULL`, which flips inside a `NOT`. A rule that is right under
a filter and wrong under a negation is worse than no rule.

The `lpad`/`rpad` collapse rides along because it is the same observation about widths:
padding to a width that has already been reached is the identity, so the outer call in
`lpad(lpad(s, 8, '0'), 8, '0')` is pure cost.
"""

from __future__ import annotations

from collections.abc import Callable

from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rule import Phase, node_rule
from batcher.kyber.rules.leaf_rewrite import rewrite_node
from batcher.plan.expr_ir import Binary, Expr, Lit
from batcher.plan.expr_ir.func_nodes import StrFunc
from batcher.plan.ir_tags import COMPARISON_FLIP
from batcher.plan.logical import Aggregate, Filter, Project, Sort, Window

__all__ = ["LENGTH_EMPTINESS_RULES", "PAD_COLLAPSE_RULES"]

_NODES = (Filter, Project, Aggregate, Sort, Window)

#: The three length functions, all of which are zero exactly for the empty string.
#: `len` counts characters, `octet_length` bytes, `bit_length` bits — the *scale* differs
#: and the zero does not, which is the only thing these rules depend on.
_LENGTH_FNS = ("len", "octet_length", "bit_length")

#: `(operator, literal) -> comparison against the empty string`. `> 0`, `>= 1` and `<> 0`
#: all mean non-empty; `<= 0` and `< 1` mean empty. `= 0` is absent for `len` alone,
#: where `exprs/text` already registers `len_zero_to_empty_string`.
_EMPTINESS: dict[tuple[str, int], str] = {
    ("gt", 0): "ne",
    ("ge", 1): "ne",
    ("ne", 0): "ne",
    ("le", 0): "eq",
    ("lt", 1): "eq",
    ("eq", 0): "eq",
}

_SUFFIX = {
    ("gt", 0): "gt_zero",
    ("ge", 1): "ge_one",
    ("ne", 0): "ne_zero",
    ("le", 0): "le_zero",
    ("lt", 1): "lt_one",
    ("eq", 0): "eq_zero",
}


def _comparison_against_int(expr: Expr) -> tuple[str, Expr, int] | None:
    if not isinstance(expr, Binary) or expr.op not in COMPARISON_FLIP:
        return None
    for computed, other, op in (
        (expr.left, expr.right, expr.op),
        (expr.right, expr.left, COMPARISON_FLIP[expr.op]),
    ):
        if (
            isinstance(other, Lit)
            and isinstance(other.value, int)
            and not isinstance(other.value, bool)
        ):
            return op, computed, other.value
    return None


def _emptiness_leaf(fn: str, key: tuple[str, int]) -> Callable[[Expr], Expr]:
    def leaf(expr: Expr) -> Expr:
        parts = _comparison_against_int(expr)
        if parts is None:
            return expr
        op, computed, value = parts
        if (op, value) != key:
            return expr
        if not isinstance(computed, StrFunc) or computed.fn != fn:
            return expr
        return Binary(_EMPTINESS[key], computed.input, Lit(""))

    return leaf


def _register(
    name: str,
    leaf: Callable[[Expr], Expr],
    expr_matches: tuple[type, ...],
    expr_ops: tuple[str, ...] | None = None,
):
    return DEFAULT_REGISTRY.add(
        node_rule(
            name,
            Phase.NORMALIZE,
            lambda node, _ctx, _leaf=leaf: rewrite_node(node, _leaf),
            matches=_NODES,
            expr_fn=leaf,
            expr_matches=expr_matches,
            expr_ops=expr_ops,
        )
    )


def _emptiness_keys(fn: str) -> list[tuple[str, int]]:
    """The comparisons this function registers. `len(s) = 0` is already covered by
    `len_zero_to_empty_string`, so it is skipped for `len` only."""
    return [key for key in _EMPTINESS if not (fn == "len" and key == ("eq", 0))]


#: Seventeen rules: five comparisons for `length` and six each for `octet_length` and
#: `bit_length`, every one of them landing on `s = ''` or `s <> ''`.
LENGTH_EMPTINESS_RULES = [
    _register(
        f"{fn}_{_SUFFIX[key]}_to_emptiness_test",
        _emptiness_leaf(fn, key),
        (Binary,),
        (key[0], COMPARISON_FLIP[key[0]]),
    )
    for fn in _LENGTH_FNS
    for key in _emptiness_keys(fn)
]


def _pad_leaf(fn: str) -> Callable[[Expr], Expr]:
    def leaf(expr: Expr) -> Expr:
        if not isinstance(expr, StrFunc) or expr.fn != fn:
            return expr
        inner = expr.input
        if (
            isinstance(inner, StrFunc)
            and inner.fn == fn
            and inner.start == expr.start
            and inner.pattern == expr.pattern
        ):
            return inner
        return expr

    return leaf


#: `lpad(lpad(s, n, f), n, f)` -> `lpad(s, n, f)`, and the `rpad` twin. Padding to a width
#: is idempotent: the inner call already produced a string of exactly `n` characters, and
#: padding *or truncating* that to `n` again returns it unchanged. The fill character and
#: the width must match, since padding to two different widths is two different results.
PAD_COLLAPSE_RULES = [
    _register(f"collapse_idempotent_{fn}", _pad_leaf(fn), (StrFunc,)) for fn in ("lpad", "rpad")
]
