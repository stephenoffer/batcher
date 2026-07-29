"""String folds whose function takes an argument beyond its input.

The siblings in `plain` fold a call that carries only its operand. These carry a
pattern, an index, or a replacement too, which adds a second way to decline: an
argument the fold does not reproduce (an uneven `translate` map, an out-of-range
`split_part` index, a negative `right` count) is left for the data plane.

Each was checked against the engine value-by-value before being written, and against
DuckDB where it has the same function. They are ASCII-guarded wherever the operation
indexes or compares characters; the byte-exact `base64` needs no guard.
"""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.kyber.rules.exprs.text_folds.literals import (
    ascii_text,
    fold,
    literal_text,
)
from batcher.kyber.rules.leaf_rewrite import rewrite_node
from batcher.plan.expr_ir.func_nodes import StrFunc
from batcher.plan.logical import Filter, LogicalPlan, Project

__all__ = [
    "fold_base64_of_literal",
    "fold_contains_of_literal",
    "fold_ends_with_of_literal",
    "fold_position_of_literal",
    "fold_right_of_literal",
    "fold_split_part_of_literal",
    "fold_starts_with_of_literal",
    "fold_translate_of_literal",
]

#
# Each of these was checked against the engine value-by-value before being written, the
# same discipline as the group above. They are ASCII-guarded wherever the operation
# indexes or compares characters, because a multi-byte operand is exactly where Python's
# code-point view and Rust's could part company; the byte-exact `base64` needs no guard.


def _base64(expr: StrFunc):
    text = literal_text(expr.input)
    if text is None:
        return None
    import base64 as _b64

    return _b64.b64encode(text.encode()).decode()


def _right(expr: StrFunc):
    text = ascii_text(expr.input)
    n = expr.start
    if text is None or n is None or n < 0:
        return None
    return text[-n:] if n else ""


def _position(expr: StrFunc):
    text = ascii_text(expr.input)
    needle = expr.pattern
    if text is None or needle is None:
        return None
    # 1-based, and 0 for "not found" -- the convention the engine and DuckDB share.
    return text.find(needle) + 1


def _predicate(op):
    def compute(expr: StrFunc):
        text = ascii_text(expr.input)
        needle = expr.pattern
        if text is None or needle is None:
            return None
        return op(text, needle)

    return compute


def _translate(expr: StrFunc):
    text = ascii_text(expr.input)
    src, dst = expr.pattern, expr.replacement
    if text is None or src is None or dst is None or len(src) != len(dst):
        return None
    return text.translate(str.maketrans(src, dst))


def _split_part(expr: StrFunc):
    text = ascii_text(expr.input)
    sep, idx = expr.pattern, expr.start
    if text is None or not sep or idx is None or idx < 1:
        return None
    parts = text.split(sep)
    # Out of range is engine-defined; only fold the in-range case.
    return parts[idx - 1] if idx <= len(parts) else None


_BASE64 = fold("base64", _base64)
_RIGHT = fold("right", _right)
_POSITION = fold("position", _position)
_CONTAINS = fold("contains", _predicate(lambda s, n: n in s))
_STARTS_WITH = fold("starts_with", _predicate(str.startswith))
_ENDS_WITH = fold("ends_with", _predicate(str.endswith))
_TRANSLATE = fold("translate", _translate)
_SPLIT_PART = fold("split_part", _split_part)


@rule(
    name="fold_base64_of_literal",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_BASE64,
    expr_matches=(StrFunc,),
)
def fold_base64_of_literal(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`base64('abc') -> 'YWJj'`. Base64 is a byte-exact encoding of the UTF-8 bytes with
    one right answer, so no ASCII guard is needed -- verified against Python's `base64`."""
    return rewrite_node(node, _BASE64)


@rule(
    name="fold_right_of_literal",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_RIGHT,
    expr_matches=(StrFunc,),
)
def fold_right_of_literal(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`right('abcdef', 3) -> 'def'`. ASCII-guarded: the count is in characters, and a
    negative count has engine-defined behaviour this fold does not reproduce."""
    return rewrite_node(node, _RIGHT)


@rule(
    name="fold_position_of_literal",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_POSITION,
    expr_matches=(StrFunc,),
)
def fold_position_of_literal(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`position('abcdef', 'cd') -> 3`. One-based, and zero when the needle is absent --
    the convention the engine shares with DuckDB, confirmed against both."""
    return rewrite_node(node, _POSITION)


@rule(
    name="fold_contains_of_literal",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_CONTAINS,
    expr_matches=(StrFunc,),
)
def fold_contains_of_literal(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`contains('abcdef', 'cd') -> TRUE`. A membership test between two constants is a
    constant. No null case can arise -- both operands are literals."""
    return rewrite_node(node, _CONTAINS)


@rule(
    name="fold_starts_with_of_literal",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_STARTS_WITH,
    expr_matches=(StrFunc,),
)
def fold_starts_with_of_literal(
    node: Filter | Project, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`starts_with('abcdef', 'ab') -> TRUE`, the prefix sibling of the `contains` fold."""
    return rewrite_node(node, _STARTS_WITH)


@rule(
    name="fold_ends_with_of_literal",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_ENDS_WITH,
    expr_matches=(StrFunc,),
)
def fold_ends_with_of_literal(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`ends_with('abcdef', 'ef') -> TRUE`, the suffix sibling of the `contains` fold."""
    return rewrite_node(node, _ENDS_WITH)


@rule(
    name="fold_translate_of_literal",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_TRANSLATE,
    expr_matches=(StrFunc,),
)
def fold_translate_of_literal(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`translate('abc', 'ab', 'xy') -> 'xyc'`. Folded only when the two character sets
    are the same length -- an uneven pair means deletion, whose semantics vary between
    engines and which this does not reproduce. ASCII-guarded, since the mapping is
    per-character."""
    return rewrite_node(node, _TRANSLATE)


@rule(
    name="fold_split_part_of_literal",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_SPLIT_PART,
    expr_matches=(StrFunc,),
)
def fold_split_part_of_literal(
    node: Filter | Project, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`split_part('a-b-c', '-', 2) -> 'b'`. One-based, and only folded when the index is
    in range: an out-of-range part is engine-defined and left alone."""
    return rewrite_node(node, _SPLIT_PART)
