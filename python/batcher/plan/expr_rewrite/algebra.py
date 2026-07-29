"""Boolean-connective algebra, column substitution, and window hoisting.

The shape-level helpers rules reach for constantly: flattening and rebuilding `AND`/`OR`
chains, inlining a column's defining expression, and lifting a window function out of an
expression into its own synthetic column.
"""

from __future__ import annotations

from collections.abc import Sequence

from batcher.plan.expr_ir import Binary, Col, Expr, WindowExpr
from batcher.plan.expr_rewrite.traverse import transform_expr_up

__all__ = [
    "WINDOW_TEMP_PREFIX",
    "combine_conjuncts",
    "combine_disjuncts",
    "hoist_windows",
    "is_bare_window",
    "split_conjuncts",
    "split_disjuncts",
    "substitute_columns",
]

# Prefix of the synthetic column a hoisted window lands in. Chosen to be un-typeable
# as a user column (leading dunder-ish underscores) so a hoist can never shadow one.
WINDOW_TEMP_PREFIX = "__bt_win_"


def split_conjuncts(expr: Expr) -> list[Expr]:
    """Flatten a top-level `AND` chain into its conjuncts (a non-AND yields `[expr]`).

    The inverse of `combine_conjuncts`. Used by predicate pushdown and predicate
    inference to reason about each conjunct independently."""
    if isinstance(expr, Binary) and expr.op == "and":
        return split_conjuncts(expr.left) + split_conjuncts(expr.right)
    return [expr]


def combine_conjuncts(exprs: list[Expr]) -> Expr:
    """Combine a non-empty list of expressions into a **balanced** `AND` tree.

    The inverse of `split_conjuncts`. The tree is balanced — depth O(log n) rather than
    the naive left-deep O(n) — so a long predicate (a fused chain of hundreds of filters,
    a large `IN` list, a generated boolean) never nests deep enough to exceed the engine's
    recursion limit when the IR is deserialized in the data plane, nor Python's own limit
    when `split_conjuncts` walks it back. `AND` is associative + commutative, so balancing
    preserves the predicate exactly (the conjuncts' left-to-right order is kept). Raises on
    an empty list (there is no neutral predicate to return without inventing a literal)."""
    if not exprs:
        raise ValueError("combine_conjuncts requires at least one expression")
    while len(exprs) > 1:
        # Pairwise-fold one level at a time (a bottom-up balanced tree); an odd tail
        # carries forward. log2(n) passes ⇒ a tree of depth ceil(log2(n)).
        exprs = [
            Binary("and", exprs[i], exprs[i + 1]) if i + 1 < len(exprs) else exprs[i]
            for i in range(0, len(exprs), 2)
        ]
    return exprs[0]


def split_disjuncts(expr: Expr) -> list[Expr]:
    """Flatten a top-level `OR` chain into its disjuncts (a non-OR yields `[expr]`).

    The inverse of `combine_disjuncts`; the `OR` analogue of `split_conjuncts`, used to
    factor a conjunct common to every branch of a disjunction out of the `OR`."""
    if isinstance(expr, Binary) and expr.op == "or":
        return split_disjuncts(expr.left) + split_disjuncts(expr.right)
    return [expr]


def combine_disjuncts(exprs: list[Expr]) -> Expr:
    """Combine a non-empty list of expressions into a left-deep `OR` chain.

    The inverse of `split_disjuncts`; raises on an empty list (no neutral disjunct
    exists without inventing a literal)."""
    if not exprs:
        raise ValueError("combine_disjuncts requires at least one expression")
    out = exprs[0]
    for e in exprs[1:]:
        out = Binary("or", out, e)
    return out


def substitute_columns(expr: Expr, mapping: dict[str, Expr]) -> Expr:
    """Replace every `Col(name)` in `expr` whose `name` is in `mapping` with the
    mapped expression. Used to rewrite a predicate/expression expressed over an
    operator's *output* columns into one over its *input* (e.g. inlining a
    projection's or a group key's defining expression when pushing a filter down)."""

    def sub(e: Expr) -> Expr:
        if isinstance(e, Col) and e.name in mapping:
            return mapping[e.name]
        return e

    return transform_expr_up(expr, sub)


def is_bare_window(expr: Expr) -> bool:
    """True when `expr` is a window whose argument holds no further window.

    Such a window needs no surrounding `Project`: it can be named directly by a
    `Window` node. `Dataset.with_columns` takes that shortcut; anything else goes
    through `hoist_windows`.
    """
    if not isinstance(expr, WindowExpr):
        return False
    return expr.input is None or not _contains_window(expr.input)


def _contains_window(expr: Expr) -> bool:
    found = False

    def probe(node: Expr) -> Expr:
        nonlocal found
        if isinstance(node, WindowExpr):
            found = True
        return node

    transform_expr_up(expr, probe)
    return found


def hoist_windows(exprs: Sequence[Expr]) -> tuple[list[Expr], list[tuple[str, WindowExpr]]]:
    """Lift every `WindowExpr` out of `exprs`, leaving a `Col` reference in its place.

    A window function has no scalar IR — the engine computes it in a relational
    `Window` operator. To let a window *compose* like a scalar
    (``col("x") - col("x").shift(1)``), the relational layer pulls each `WindowExpr`
    out into its own synthetic column and rewrites the surrounding tree to read that
    column. This function does the expression half; the caller builds one `Window`
    node per returned pair and projects the rewritten expressions on top.

    Windows nested inside a window's argument (``col("x").shift(1).cum_sum()``) are
    hoisted first, so the returned pairs are already in dependency order: building
    them front-to-back, each `Window` node sees the columns the next one reads.

    `WindowExpr` is a leaf to `transform_expr_up` (it carries no `_EXPR_KIDS` entry),
    so this recurses into its argument explicitly.

    One `WindowExpr` *object* reached from several places in the tree is hoisted once
    and shared. A builder that reuses a window — ``when(w >= n).then(w)`` — would
    otherwise emit two identical `Window` nodes and compute it twice. Identity is the
    right key (and safe): the nodes stay alive in `exprs` for the whole call, and
    without `__eq__`/`__hash__` on `Expr` there is no structural key to use.

    Args:
        exprs: The scalar expressions to rewrite.

    Returns:
        The rewritten expressions, and the ``(column_name, window)`` pairs to
        materialize before evaluating them — empty when `exprs` held no window.
    """
    hoisted: list[tuple[str, WindowExpr]] = []
    seen: dict[int, str] = {}  # id(WindowExpr) -> the column it was hoisted into

    def rule(node: Expr) -> Expr:
        if not isinstance(node, WindowExpr):
            return node
        shared = seen.get(id(node))
        if shared is not None:
            return Col(shared)
        # Recurse into the argument first: an inner window must be materialized
        # before the outer one can read it.
        inner = node if node.input is None else node.with_input(lift(node.input))
        name = f"{WINDOW_TEMP_PREFIX}{len(hoisted)}"
        seen[id(node)] = name
        hoisted.append((name, inner))
        return Col(name)

    def lift(expr: Expr) -> Expr:
        # A bare column reference cannot contain a window, and it is what almost every
        # expression in a wide projection *is*: `with_columns` re-emits one pass-through
        # `Col` per untouched column, so a 300-column relation reaches here 300 times with
        # nothing to hoist. Answering that with a type check instead of a tree rewrite is
        # what keeps the call proportional to the columns it actually changes.
        if type(expr) is Col:
            return expr
        return transform_expr_up(expr, rule)

    return [lift(e) for e in exprs], hoisted
