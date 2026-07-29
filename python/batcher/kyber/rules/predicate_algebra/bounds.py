"""Union the disjuncts a generated predicate leaves on one column.

Six rewrites, all of them collapsing `p OR q` on a single column into one comparison:

    x < 3  OR x < 7        ->  x < 7                (the weaker upper bound wins)
    x > 7  OR x >= 3       ->  x >= 3               (the weaker lower bound wins)
    x = 3  OR x > 3        ->  x >= 3               (the equality closes the bound)
    x = 3  OR x < 3        ->  x <= 3
    x IN (1, 2) OR x = 3   ->  x IN (1, 2, 3)
    (x >= 1 AND x <= 5) OR (x >= 4 AND x <= 9)  ->  x >= 1 AND x <= 9

plus the conjunction case where an equality already implies its neighbour
(`x = 5 AND x > 1` -> `x = 5`), and the constant fold `'a' IN ('a', 'b') -> true` — a shape
no query writes, but the one the `CASE` push rules produce once a membership test lands on
a literal branch.

The value is not the saved comparison. It is that each leaves *one* bound or *one* set per
column, which is the only shape `zonemap_prune_filter` and source predicate pushdown
recognize; a column mentioned by two disjuncts is skipped by both, so the query reads every
row group it could have refuted.

Comparing two bounds means comparing their literals, and that is done with a guarded
Python comparison rather than assumed. Two literals of incomparable types (an int against
a string, a date against a number) make the rule decline — the engine's coercion rules are
not re-derived here. Booleans are excluded outright: `True` compares as `1` in Python,
which would silently order a boolean column by an integer rule.

Every rule is exact under three-valued logic with no non-null guard. A null operand makes
both disjuncts `NULL`, `NULL OR NULL` is `NULL`, and the single surviving comparison on the
same operand is `NULL` too.
"""

from __future__ import annotations

from collections.abc import Callable

from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rule import Phase, node_rule
from batcher.kyber.rules.leaf_rewrite import rewrite_node
from batcher.plan.expr_ir import Binary, Expr, InList, Lit
from batcher.plan.expr_rewrite import expr_key
from batcher.plan.logical import Aggregate, Filter, Project, Sort, Window

__all__ = ["PREDICATE_BOUND_RULES"]

_NODES = (Filter, Project, Aggregate, Sort, Window)
_FLIP = {"eq": "eq", "ne": "ne", "lt": "gt", "gt": "lt", "le": "ge", "ge": "le"}
#: Which side of the number line each ordered operator bounds, and whether it includes the
#: endpoint. The inclusive flag is the tie-break when two bounds share a literal.
_UPPER = {"lt": False, "le": True}
_LOWER = {"gt": False, "ge": True}


def _scalar(expr: Expr) -> object | None:
    """The value of a non-boolean literal, else ``None``.

    Booleans are rejected because Python orders `True` as `1`, which would let an integer
    bound rule silently apply to a boolean column.
    """
    if isinstance(expr, Lit) and expr.value is not None and not isinstance(expr.value, bool):
        return expr.value
    return None


def _comparison(expr: Expr) -> tuple[str, Expr, object] | None:
    """`(op, operand, literal)` for a comparison against a literal, operand-first."""
    if not isinstance(expr, Binary) or expr.op not in _FLIP:
        return None
    for operand, other, op in (
        (expr.left, expr.right, expr.op),
        (expr.right, expr.left, _FLIP[expr.op]),
    ):
        value = _scalar(other)
        if value is not None:
            return op, operand, value
    return None


def _lt(left: object, right: object) -> bool | None:
    """`left < right`, or ``None`` when the two are not comparable."""
    try:
        return bool(left < right)  # type: ignore[operator]
    except TypeError:
        return None


def _same_operand(a: Expr, b: Expr) -> bool:
    return expr_key(a) == expr_key(b)


def _widen_bound(directions: dict[str, bool], *, upper: bool) -> Callable[[Expr], Expr]:
    """The leaf unioning two same-direction bounds in a disjunction.

    For an upper bound the union keeps the *larger* literal, and the inclusive operator
    when the two literals are equal; for a lower bound it keeps the smaller.
    """

    def leaf(expr: Expr) -> Expr:
        if not isinstance(expr, Binary) or expr.op != "or":
            return expr
        left, right = _comparison(expr.left), _comparison(expr.right)
        if left is None or right is None:
            return expr
        (left_op, left_operand, left_value) = left
        (right_op, right_operand, right_value) = right
        if left_op not in directions or right_op not in directions:
            return expr
        if not _same_operand(left_operand, right_operand):
            return expr
        less = _lt(left_value, right_value)
        if less is None:
            return expr
        if left_value == right_value:
            # Same endpoint: the inclusive operator is the weaker of the two.
            return expr.left if directions[left_op] else expr.right
        left_is_weaker = (not less) if upper else less
        return expr.left if left_is_weaker else expr.right

    return leaf


def _close_bound(directions: dict[str, bool], inclusive: str) -> Callable[[Expr], Expr]:
    """The leaf absorbing `x = c` into a bound at the same `c`, closing it.

    `x = 3 OR x > 3` is `x >= 3`, and `x = 3 OR x >= 3` is already `x >= 3`; both are
    handled by replacing the pair with the inclusive operator at that literal.
    """

    def leaf(expr: Expr) -> Expr:
        if not isinstance(expr, Binary) or expr.op != "or":
            return expr
        left, right = _comparison(expr.left), _comparison(expr.right)
        if left is None or right is None:
            return expr
        for equality, bound in ((left, right), (right, left)):
            if equality[0] != "eq" or bound[0] not in directions:
                continue
            if not _same_operand(equality[1], bound[1]) or equality[2] != bound[2]:
                continue
            return Binary(inclusive, bound[1], Lit(bound[2]))
        return expr

    return leaf


def _equality_absorbs_bound(expr: Expr) -> Expr:
    """`x = c AND x OP d` -> `x = c` when `c` already satisfies `x OP d`."""
    if not isinstance(expr, Binary) or expr.op != "and":
        return expr
    left, right = _comparison(expr.left), _comparison(expr.right)
    if left is None or right is None:
        return expr
    for equality, bound, equality_side in ((left, right, expr.left), (right, left, expr.right)):
        if equality[0] != "eq" or bound[0] not in _FLIP or bound[0] == "eq":
            continue
        if not _same_operand(equality[1], bound[1]):
            continue
        if _satisfies(equality[2], bound[0], bound[2]):
            return equality_side
    return expr


def _satisfies(value: object, op: str, bound: object) -> bool:
    """Whether the constant `value` satisfies `value OP bound`, ``False`` if uncomparable."""
    less = _lt(value, bound)
    if less is None:
        return False
    equal = value == bound
    return {
        "lt": less,
        "le": less or equal,
        "gt": not less and not equal,
        "ge": not less,
        "ne": not equal,
    }.get(op, False)


def _in_list_values(expr: Expr) -> tuple[Expr, tuple] | None:
    if isinstance(expr, InList):
        return expr.input, expr.values
    comparison = _comparison(expr)
    if comparison is not None and comparison[0] == "eq":
        return comparison[1], (comparison[2],)
    return None


def _merge_in_lists(expr: Expr) -> Expr:
    """`x IN (a, b) OR x = c` and `x IN (a, b) OR x IN (c, d)` -> one `IN` list."""
    if not isinstance(expr, Binary) or expr.op != "or":
        return expr
    if not isinstance(expr.left, InList) and not isinstance(expr.right, InList):
        return expr  # two bare equalities are `or_equalities_to_in_list`'s job
    left, right = _in_list_values(expr.left), _in_list_values(expr.right)
    if left is None or right is None or not _same_operand(left[0], right[0]):
        return expr
    merged = list(left[1]) + [v for v in right[1] if v not in left[1]]
    if len(merged) == len(left[1]) and set(merged) == set(left[1]) | set(right[1]):
        return expr.left  # the right side added nothing
    return InList(left[0], tuple(merged))


def _range(expr: Expr) -> tuple[Expr, str, object, str, object] | None:
    """`(operand, lower_op, lower, upper_op, upper)` for an `x >= a AND x <= b` shape."""
    if not isinstance(expr, Binary) or expr.op != "and":
        return None
    left, right = _comparison(expr.left), _comparison(expr.right)
    if left is None or right is None or not _same_operand(left[1], right[1]):
        return None
    for lower, upper in ((left, right), (right, left)):
        if lower[0] in _LOWER and upper[0] in _UPPER:
            return lower[1], lower[0], lower[2], upper[0], upper[2]
    return None


def _union_ranges(expr: Expr) -> Expr:
    """Collapse two overlapping ranges on one column into their union.

    The ranges must genuinely overlap or touch. Two disjoint ranges have no single-interval
    union, and widening them to their hull would admit rows neither disjunct accepted.
    """
    if not isinstance(expr, Binary) or expr.op != "or":
        return expr
    left, right = _range(expr.left), _range(expr.right)
    if left is None or right is None or not _same_operand(left[0], right[0]):
        return expr
    operand = left[0]
    _, low_op_l, low_l, high_op_l, high_l = left
    _, low_op_r, low_r, high_op_r, high_r = right
    overlaps = _overlap(low_l, high_l, low_r, high_r)
    if overlaps is not True:
        return expr
    low_op, low = _weaker_lower((low_op_l, low_l), (low_op_r, low_r))
    high_op, high = _weaker_upper((high_op_l, high_l), (high_op_r, high_r))
    return Binary("and", Binary(low_op, operand, Lit(low)), Binary(high_op, operand, Lit(high)))


def _overlap(low_l: object, high_l: object, low_r: object, high_r: object) -> bool | None:
    """Whether the two closed-ish intervals intersect, ``None`` if uncomparable.

    The endpoints' strictness is ignored on purpose: treating both as closed can only make
    the test *more* permissive at a shared endpoint, and two ranges that touch at one point
    still have a single-interval union.
    """
    first = _lt(high_l, low_r)
    second = _lt(high_r, low_l)
    if first is None or second is None:
        return None
    return not (first or second)


def _weaker_lower(a: tuple[str, object], b: tuple[str, object]) -> tuple[str, object]:
    less = _lt(a[1], b[1])
    if less is None or a[1] == b[1]:
        return a if a[0] == "ge" else b
    return a if less else b


def _weaker_upper(a: tuple[str, object], b: tuple[str, object]) -> tuple[str, object]:
    less = _lt(a[1], b[1])
    if less is None or a[1] == b[1]:
        return a if a[0] == "le" else b
    return b if less else a


def _fold_in_list_of_literal(expr: Expr) -> Expr:
    """`'a' IN ('a', 'b') -> true`. A membership test over a constant input is a constant.

    The shape is not something a query writes; it is what the `CASE` push rules produce.
    `(CASE WHEN c THEN 1 ELSE 3 END) IN (1, 2)` becomes an `IN` test per branch, and each
    branch value is a literal — so without this fold the push replaces one membership test
    with two and the chain stops there. With it, every branch becomes a boolean constant
    and the surrounding `CASE` rules collapse the whole expression.
    """
    if isinstance(expr, InList) and isinstance(expr.input, Lit) and expr.input.value is not None:
        return Lit(expr.input.value in expr.values)
    return expr


def _register(name: str, leaf: Callable[[Expr], Expr]):
    return DEFAULT_REGISTRY.add(
        node_rule(
            name,
            Phase.NORMALIZE,
            lambda node, _ctx, _leaf=leaf: rewrite_node(node, _leaf),
            matches=_NODES,
            expr_fn=leaf,
            expr_matches=(Binary, InList),
            # Every leaf here reads a *connective* -- it absorbs, widens, or unions the two
            # sides of one `AND`/`OR`. A comparison Binary is only ever its operand, reached
            # by recursion rather than offered to the leaf. `InList` carries no operator, so
            # the one leaf that targets it is unaffected by this filter.
            expr_ops=("and", "or"),
        )
    )


#: The eight disjunction/absorption/folding rules this module registers, in the order the
#: module docstring lists them.
PREDICATE_BOUND_RULES = [
    _register("widen_upper_bound_disjunction", _widen_bound(_UPPER, upper=True)),
    _register("widen_lower_bound_disjunction", _widen_bound(_LOWER, upper=False)),
    _register("close_upper_bound_with_equality", _close_bound(_UPPER, "le")),
    _register("close_lower_bound_with_equality", _close_bound(_LOWER, "ge")),
    _register("equality_absorbs_implied_bound", _equality_absorbs_bound),
    _register("merge_in_list_disjunction", _merge_in_lists),
    _register("union_overlapping_range_disjunction", _union_ranges),
    _register("fold_in_list_of_literal_input", _fold_in_list_of_literal),
]
