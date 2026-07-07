"""Structural join rewrites — collapse a join whose result is provably fixed.

The rules here fire on the *shape* of a join, never on statistics. Two families:

* **Empty-side elimination.** A join with a structurally-empty input (the canonical
  ``Limit(x, 0)`` marker the pruning rules and ``.limit(0)`` produce) often has a
  provable result: an inner or semi join to an empty side is empty; an anti join to
  an empty side keeps its left rows; an outer join whose *null-supplying* side is
  empty degenerates to its preserved side. These rewrite the join to an empty
  relation or a projection of a single side — but only when the output schema can be
  carried by one existing input. The engine's IR has no null literal and no
  values/empty-schema node, so a two-sided output schema (a general inner/outer join
  whose columns come from *both* sides) cannot be fabricated as a zero-row relation;
  those cases are deliberately left untouched (see the module notes).

* **Key canonicalization.** ``dedup_join_keys`` drops a duplicated equi-key pair
  (``k = c AND k = c`` is ``k = c``), a pure normalization that never changes the
  result and unblocks the single-key fast path.

Every rule returns a non-``Join`` node (a ``Project`` or ``Limit``) or a key-reduced
join and returns ``None`` once there is nothing to do, so each is idempotent and the
fixpoint terminates. Correctness is the whole point: when a case is not provably
safe (mixed-side output, a preserved side that is the empty one, NULL-key subtleties)
the rule returns ``None``.
"""

from __future__ import annotations

import dataclasses

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.plan.expr_ir import Col
from batcher.plan.logical import (
    Join,
    JoinOutputCol,
    Limit,
    LogicalPlan,
    Project,
    Projection,
)

__all__ = [
    "anti_join_empty_right",
    "dedup_join_keys",
    "inner_join_empty_to_empty",
    "preserving_join_empty_null_side",
    "semi_anti_join_empty_left",
    "semi_join_empty_right",
]

# Which side(s) an outer join preserves — the rows it keeps even without a match.
# The empty-side rewrites may only reduce a join to a side that is *preserved* (so
# every one of that side's rows survives) while the *other* side is empty.
_PRESERVED_SIDES = {
    "left": frozenset({"left"}),
    "right": frozenset({"right"}),
    "full": frozenset({"left", "right"}),
}


def _is_empty(node: LogicalPlan) -> bool:
    """Whether `node` provably yields zero rows (the `Limit(_, 0)` empty marker)."""
    return isinstance(node, Limit) and node.n == 0


def _output_side(output: tuple[JoinOutputCol, ...]) -> str | None:
    """The single side all output columns are drawn from, or `None` if mixed/empty.

    An empty-side rewrite can only carry the join's output schema on one existing
    input; that is possible exactly when every output column comes from one side.
    """
    sides = {o.side for o in output}
    return next(iter(sides)) if len(sides) == 1 else None


def _passthrough(side_plan: LogicalPlan, output: tuple[JoinOutputCol, ...]) -> Project | None:
    """`Project(side_plan, …)` reproducing `output` by reading each column from
    `side_plan` under its source name and re-binding the join's output alias.

    Returns `None` if any source column is not present on `side_plan` (a defensive
    guard — never rewrite against a schema we cannot reproduce).
    """
    available = set(side_plan.available_columns())
    if any(o.name not in available for o in output):
        return None
    return Project(side_plan, tuple(Projection(o.alias, Col(o.name)) for o in output))


@rule(name="semi_anti_join_empty_left", phase=Phase.REWRITE, matches=(Join,))
def semi_anti_join_empty_left(node: Join, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Semi/Anti Join(∅, R)` → an empty relation with the left schema.

    A semi or anti join only ever emits (a subset of) its left rows, so an empty left
    input yields an empty result regardless of the right side or the join kind. The
    left is already the `Limit(_, 0)` empty marker, and a semi/anti join's output is
    left-only, so projecting that empty input under the output aliases reproduces the
    exact result schema with zero rows. Returns None when the left is not provably
    empty; the rewritten node is a `Project`, so the rule fires at most once.
    """
    if node.join_type not in ("semi", "anti") or not _is_empty(node.left):
        return None
    return _passthrough(node.left, node.output)


@rule(name="semi_join_empty_right", phase=Phase.REWRITE, matches=(Join,))
def semi_join_empty_right(node: Join, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Semi Join(L, ∅)` → an empty relation with the left schema.

    A semi join keeps a left row only if its key has *some* match on the right; an
    empty right side can match nothing, so the result is empty. The output is
    left-only, so a zero-row projection of the left input carries the right schema:
    `Limit(Project(L, out), 0)`. Returns None when the right is not provably empty
    (and the `Limit`/`Project` result is not a `Join`, so it never re-fires).
    """
    if node.join_type != "semi" or not _is_empty(node.right):
        return None
    projected = _passthrough(node.left, node.output)
    return None if projected is None else Limit(projected, 0)


@rule(name="anti_join_empty_right", phase=Phase.REWRITE, matches=(Join,))
def anti_join_empty_right(node: Join, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Anti Join(L, ∅)` → `L` (every left row survives).

    An anti join emits each left row whose key has *no* match on the right; against an
    empty right side nothing matches, so every left row survives unchanged. The
    output is left-only, so the result is just the left input re-projected under the
    output aliases. Returns None when the right is not provably empty. (Distinct from
    an empty *left*, which `semi_anti_join_empty_left` handles.)
    """
    if node.join_type != "anti" or not _is_empty(node.right):
        return None
    return _passthrough(node.left, node.output)


@rule(name="dedup_join_keys", phase=Phase.REWRITE, matches=(Join,))
def dedup_join_keys(node: Join, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Drop a duplicated equi-key pair from a join's key list.

    Repeating an equality in the join condition is redundant — `a = c AND a = c` is
    exactly `a = c`, including under NULLs (a null key fails the equality either way),
    so removing the duplicate `(left_key, right_key)` pair cannot change the result.
    Only *identical* pairs are removed; `(a, c)` and `(b, c)` express different
    constraints and both stay. The win is unblocking the single-key hash path and
    cleaner cost estimates. Fires only when a genuine duplicate exists (so the rule is
    idempotent); a pair repeated three times collapses over successive passes.
    """
    if len(node.left_keys) < 2:
        return None
    seen: set[tuple[str, str]] = set()
    left_keys: list[str] = []
    right_keys: list[str] = []
    for lk, rk in zip(node.left_keys, node.right_keys, strict=True):
        if (lk, rk) in seen:
            continue
        seen.add((lk, rk))
        left_keys.append(lk)
        right_keys.append(rk)
    if len(left_keys) == len(node.left_keys):
        return None  # no duplicate pair → nothing to do
    return dataclasses.replace(node, left_keys=tuple(left_keys), right_keys=tuple(right_keys))


@rule(name="inner_join_empty_to_empty", phase=Phase.PUSHDOWN, matches=(Join,))
def inner_join_empty_to_empty(node: Join, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Inner Join` with an empty input → an empty relation, when the output is single-sided.

    An inner join emits a row only when both sides match, so an empty input on either
    side makes the whole result empty. Representing that empty result needs a plan
    carrying the join's output schema; that is only possible when every output column
    comes from one side (which column pruning produces once a query reads a single
    side of the join). The result is `Limit(Project(<output side>, out), 0)`. A join
    whose output still spans both sides is left untouched — the IR cannot fabricate a
    two-sided zero-row schema. Idempotent (the result is not a `Join`).
    """
    if node.join_type != "inner" or not (_is_empty(node.left) or _is_empty(node.right)):
        return None
    side = _output_side(node.output)
    if side is None:
        return None
    src = node.left if side == "left" else node.right
    projected = _passthrough(src, node.output)
    return None if projected is None else Limit(projected, 0)


@rule(name="preserving_join_empty_null_side", phase=Phase.PUSHDOWN, matches=(Join,))
def preserving_join_empty_null_side(node: Join, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Left/Right/Full Join` whose empty side is the null-supplying one → the preserved side.

    An outer join keeps every row of its preserved side(s); the other side only
    contributes matches (and, for a `full` join, its own unmatched rows). When that
    other side is empty *and* the query reads only the preserved side's columns, the
    join degenerates to each preserved-side row exactly once — no fan-out (nothing to
    match) and no null-extended rows to fabricate — i.e. a projection of the preserved
    side. Fires only when: the output is entirely one side, that side is a *preserved*
    side of this join type, and the *other* side is provably empty. Anything else
    (the preserved side itself empty, or output columns from the empty side) would
    need null columns the IR cannot express, so it returns None. Idempotent (the
    result is a `Project`).
    """
    preserved = _PRESERVED_SIDES.get(node.join_type)
    if preserved is None:  # inner / semi / anti handled elsewhere
        return None
    side = _output_side(node.output)
    if side is None or side not in preserved:
        return None
    other = node.right if side == "left" else node.left
    if not _is_empty(other):
        return None
    src = node.left if side == "left" else node.right
    return _passthrough(src, node.output)
