"""Rewrite a cartesian join plus an inequality filter into a `RangeJoin`.

``A JOIN B ON a.x < b.y`` has no equality to co-partition on, so it lowers to an inner
join on the synthetic constant ``__cross_key`` with the predicate as a `Filter` above.
That is correct and it is what `tests/differential/test_diff_theta_join.py` pins — but
the cross product is *materialized* before the filter sees it, so the intermediate is
``|A| x |B|`` rows however few survive. It is quadratic in time and in memory, and it
stops running entirely at sizes DuckDB handles comfortably.

This rule moves the inequalities *into* the join, where the engine executes them with an
output-sensitive algorithm (a sorted-suffix scan for one, IEJoin for two). Nothing else
about the plan changes: conjuncts the rule does not consume stay in the `Filter`, so the
rewrite is a restriction of the cartesian plan rather than a reinterpretation of it.

It declines any predicate carrying an equi-conjunct across the join, because an equality
is worth more than an inequality: absorbed into the join keys by `derive_join_keys` it
makes the whole thing a hash join, which beats any range algorithm.

An operand that is *computed* rather than a bare column — ``a.ts - 5 < b.ts``, the canonical
temporal proximity join — is materialized as a hidden column beneath the join, so the shape
that motivates range joins in the first place is not the one shape that misses them. Only
non-raising expressions qualify; see `_HiddenKeys`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase

# The one non-raising-expression whitelist, shared rather than restated: both rules move a
# computation onto a join input, and both are unsound for an expression that can raise.
from batcher.kyber.rules.joins.projection import _is_push_safe
from batcher.plan.expr_ir import Binary, Col, Expr, referenced_columns
from batcher.plan.expr_rewrite import combine_conjuncts, split_conjuncts, substitute_columns
from batcher.plan.ir_tags import ORDERING_COMPARISONS, ORDERING_FLIP
from batcher.plan.logical import (
    Filter,
    Join,
    LogicalPlan,
    Project,
    Projection,
    RangeCondition,
    RangeJoin,
    is_cartesian_key_pair,
)
from batcher.plan.schema import SchemaRef
from batcher.plan.types import infer_type

__all__ = ["derive_range_join"]

# The four inequalities the engine's range join understands, and the same comparison read
# from the other side (`b.y > a.x` is `a.x < b.y`).

# IEJoin sorts on two axes, so two inequalities is the ceiling. A third stays in the
# filter above, where it is a cheap post-check on the surviving pairs rather than a
# predicate over the whole cartesian product.
_MAX_CONDITIONS = 2


@rule(name="derive_range_join", phase=Phase.PUSHDOWN, matches=(Filter,))
def derive_range_join(node: Filter, ctx: OptimizerContext) -> LogicalPlan | None:
    """Turn `Filter(cartesian Join)` with inequality conjuncts into a `RangeJoin`.

    Fires only on an inner join whose every key pair is a cartesian pseudo-edge — the
    shape a theta join lowers to. A join already driven by a real key is a hash join and
    is left alone.

    Args:
        node: The candidate `Filter`.
        ctx: The optimizer context; the rewrite is recorded on `ctx.notes`.

    Returns:
        The rewritten plan, or `None` when the shape does not apply.
    """
    join = node.input
    if not isinstance(join, Join) or join.join_type != "inner":
        return None
    if not join.left_keys:
        return None
    if not all(
        is_cartesian_key_pair(join.left, lk, join.right, rk)
        for lk, rk in zip(join.left_keys, join.right_keys, strict=True)
    ):
        return None

    left_src = {o.alias: o.name for o in join.output if o.side == "left"}
    right_src = {o.alias: o.name for o in join.output if o.side == "right"}

    conjuncts = list(split_conjuncts(node.predicate))
    # An equality across the join is strictly more valuable: `derive_join_keys` turns it
    # into a real join key and the whole thing becomes a hash join. Defer unconditionally
    # rather than depend on rule ordering to get this right.
    if any(_crossing_pair(c, left_src, right_src, ("eq",)) for c in conjuncts):
        return None

    left_schema, right_schema = _schemas(join)
    hidden = _HiddenKeys(join, left_schema, right_schema)
    candidates: list[tuple[Expr, _Candidate]] = []
    keep: list[Expr] = []
    for conj in conjuncts:
        cand = _candidate(conj, left_src, right_src, hidden, left_schema, right_schema)
        if cand is None:
            keep.append(conj)
        else:
            candidates.append((conj, cand))
    chosen = _choose(candidates)
    if not chosen:
        return None
    # Identity, not equality: two structurally equal conjuncts are distinct entries and
    # dropping both because one was chosen would delete a predicate.
    picked = {id(conj) for conj, _ in chosen}
    keep.extend(conj for conj, _ in candidates if id(conj) not in picked)

    # Only now materialize: `_HiddenKeys.add` is a side effect, and a candidate `_choose`
    # passed over must not leave a computed column behind that nothing reads.
    conditions = [cand.realize(hidden) for _, cand in chosen]

    result: LogicalPlan = RangeJoin(
        hidden.left_input(), hidden.right_input(), tuple(conditions), "inner", join.output
    )
    if keep:
        result = Filter(result, combine_conjuncts(keep))
    ctx.notes.setdefault("range_joins", []).append(
        " AND ".join(f"{c.left_key} {c.op} {c.right_key}" for c in conditions)
    )
    return result


def _crossing_pair(
    conj: Expr,
    left_src: dict[str, str],
    right_src: dict[str, str],
    ops: tuple[str, ...],
) -> tuple[str, str, str] | None:
    """`(left_key, right_key, op)` if `conj` compares a left column to a right one.

    Both operands must be bare columns, one resolving to a left output alias and the
    other to a right one. The op is returned oriented ``left OP right``, flipped when the
    operands arrive the other way round. Returned names are the *source* column names on
    each input, which is the vocabulary a join node's keys are phrased in.
    """
    if not isinstance(conj, Binary) or conj.op not in ops:
        return None
    lhs, rhs = conj.left, conj.right
    if not (isinstance(lhs, Col) and isinstance(rhs, Col)):
        return None
    if lhs.name in left_src and rhs.name in right_src:
        return (left_src[lhs.name], right_src[rhs.name], conj.op)
    if lhs.name in right_src and rhs.name in left_src:
        return (left_src[rhs.name], right_src[lhs.name], ORDERING_FLIP.get(conj.op, conj.op))
    return None


def _schemas(join: Join) -> tuple[SchemaRef | None, SchemaRef | None]:
    """Both inputs' schemas, or `None` for either side that cannot be inferred."""
    try:
        return join.left.available_schema(), join.right.available_schema()
    except Exception:  # pragma: no cover - an un-inferable schema is not an error
        return None, None


def _keys_share_a_type(
    left_schema: SchemaRef | None,
    right_schema: SchemaRef | None,
    left_key: str,
    right_key: str,
) -> bool:
    """Whether the two keys are known to have the same Arrow type.

    Unknown is treated as *not* shared: without a proof the rewrite would be trading a
    slow-but-correct plan for one the engine may refuse to run.
    """
    if left_schema is None or right_schema is None:
        return False
    if not (left_schema.has(left_key) and right_schema.has(right_key)):
        return False
    return left_schema.field(left_key).type.equals(right_schema.field(right_key).type)


class _HiddenKeys:
    """Materializes a computed range-join operand as a hidden column on the side it reads.

    A range join sorts its key *columns*, so `a.ts - 5 < b.ts` — the canonical temporal
    proximity join, and the shape `a.x + 1 < b.y` shares — has nothing to sort unless the
    expression is computed first. Computing it in a `Project` beneath the join is the same
    per-row work the filter above the cartesian product was already doing, on the same rows,
    so it is a rewrite rather than an approximation.

    "On the same rows" is the load-bearing part, and it is why only *non-raising*
    expressions qualify (`_is_push_safe`, shared with `push_projection_through_join` rather
    than restated). If the other side is empty the cartesian product is empty and the filter
    never runs, so an expression that can raise would raise here where the old plan returned
    an empty relation.
    """

    def __init__(self, join: Join, left_schema: SchemaRef | None, right_schema: SchemaRef | None):
        self._join = join
        self._schemas = {"left": left_schema, "right": right_schema}
        self._items: dict[str, list[Projection]] = {"left": [], "right": []}
        self._taken = {
            "left": set(join.left.available_columns()),
            "right": set(join.right.available_columns()),
        }

    def schema_of(self, side: str) -> SchemaRef | None:
        """That side's input schema, or `None` when it cannot be inferred."""
        return self._schemas[side]

    def type_of(self, side: str, expr: Expr) -> pa.DataType | None:
        """The Arrow type `expr` produces over `side`'s input schema, or `None` if unknown."""
        schema = self._schemas[side]
        return None if schema is None else infer_type(expr, schema)

    def add(self, side: str, expr: Expr) -> str:
        """Materialize `expr` on `side` and return the hidden column's name."""
        name = "__rj_key"
        while name in self._taken[side]:
            name += "_"
        self._taken[side].add(name)
        self._items[side].append(Projection(name, expr))
        return name

    def left_input(self) -> LogicalPlan:
        return self._wrap(self._join.left, self._items["left"])

    def right_input(self) -> LogicalPlan:
        return self._wrap(self._join.right, self._items["right"])

    @staticmethod
    def _wrap(plan: LogicalPlan, extra: list[Projection]) -> LogicalPlan:
        if not extra:
            return plan
        passthrough = [Projection(c, Col(c)) for c in plan.available_columns()]
        return Project(plan, (*passthrough, *extra))


@dataclass(frozen=True)
class _Candidate:
    """One inequality the rule *could* move into the join, before it commits to doing so.

    Separating "can this be a join condition" from "is this one of the two we keep" is what
    lets [`_choose`] pick a pair, since materializing a computed operand
    ([`_HiddenKeys.add`]) is a side effect that must not happen for a condition the choice
    passes over. Entry 13 of the improvements ledger is the reason that seam exists: it is
    what made a plausible-sounding selection heuristic testable, and therefore refutable.

    `left_expr`/`right_expr` are set only for the computed case, and hold the expression
    already rewritten into that side's *source* column names.
    """

    left_key: str | None
    right_key: str | None
    op: str
    left_expr: Expr | None = None
    right_expr: Expr | None = None

    def realize(self, hidden: _HiddenKeys) -> RangeCondition:
        """Commit: materialize any computed operand and return the wire condition."""
        left = self.left_key if self.left_expr is None else hidden.add("left", self.left_expr)
        right = self.right_key if self.right_expr is None else hidden.add("right", self.right_expr)
        assert left is not None and right is not None
        return RangeCondition(left, right, self.op)


def _choose(candidates: list[tuple[Expr, _Candidate]]) -> list[tuple[Expr, _Candidate]]:
    """Pick at most two candidates. **Written order**, and that is a considered choice.

    IEJoin sorts on two axes and a 2-D overlap has four crossing inequalities, so something
    has to choose. The obvious heuristic — take one condition per *dimension*, on the theory
    that two constraints on the same dimension are redundant — is wrong, and measurably so.
    For a bounding-box overlap the two x conditions together express *x-overlap*, which for
    boxes of side `w` in a range `R` selects about `2w/R` of all pairs; one x condition plus
    one y condition selects about `1/4`, because a lone inequality on an axis is barely
    selective at all. Measured over 25 million random pairs (side 80 in a range of 4,000):
    the axis pair passes **3.94%** to the filter above, the mixed pick **27.12%**, against a
    true answer of 0.156%. Seven times the intermediate, for a heuristic that sounds right.

    So the pair a user writes *adjacently* is the pair that belongs together, and written
    order preserves it. Nothing at plan time distinguishes the two cases: the true
    selectivity of an axis pair comes from the correlation between `lo` and `hi`, which an
    independence-assuming estimator cannot see, and assuming independence would pick the
    losing pair here. Left as written rather than guessed.
    """
    return candidates[:_MAX_CONDITIONS]


def _candidate(
    conj: Expr,
    left_src: dict[str, str],
    right_src: dict[str, str],
    hidden: _HiddenKeys,
    left_schema: SchemaRef | None,
    right_schema: SchemaRef | None,
) -> _Candidate | None:
    """The `_Candidate` for `conj`, or `None` when it must stay in the filter."""
    pair = _crossing_pair(conj, left_src, right_src, ORDERING_COMPARISONS)
    if pair is not None:
        # The engine encodes both sides of a condition with one row converter, so a pair
        # whose keys do not share a type cannot be joined this way.
        if not _keys_share_a_type(left_schema, right_schema, pair[0], pair[1]):
            return None
        return _Candidate(pair[0], pair[1], pair[2])
    return _computed_candidate(conj, left_src, right_src, hidden)


def _computed_candidate(
    conj: Expr,
    left_src: dict[str, str],
    right_src: dict[str, str],
    hidden: _HiddenKeys,
) -> _Candidate | None:
    """A candidate for `<expr> OP <col>` (or the reverse) across the join, or `None`.

    One operand must be a bare column of one side; the other must be an expression reading
    **only** the other side's columns, naming at least one of them, and built entirely from
    operations that cannot raise. The expression is rewritten into that side's source column
    names; nothing is materialized here.

    Both keys must still end up sharing an Arrow type, since the engine encodes them with one
    row converter — so the computed operand's inferred type is checked against the plain
    column's before the candidate is offered.
    """
    if not isinstance(conj, Binary) or conj.op not in ORDERING_COMPARISONS:
        return None
    orientations = (
        (conj.left, conj.right, conj.op),
        (conj.right, conj.left, ORDERING_FLIP[conj.op]),
    )
    for expr, other, op in orientations:
        # `other` is the plain key column; `expr` is the candidate computation, and `op` is
        # oriented so that it reads `expr OP other`.
        if not isinstance(other, Col) or isinstance(expr, Col):
            continue
        if not _is_push_safe(expr):
            return None
        refs = referenced_columns(expr)
        if not refs:
            return None
        for expr_side, other_side in (("left", "right"), ("right", "left")):
            src = left_src if expr_side == "left" else right_src
            other_map = right_src if expr_side == "left" else left_src
            if other.name not in other_map or not refs <= set(src):
                continue
            rewritten = substitute_columns(expr, {a: Col(n) for a, n in src.items()})
            computed_type = hidden.type_of(expr_side, rewritten)
            other_schema = hidden.schema_of(other_side)
            other_name = other_map[other.name]
            if (
                computed_type is None
                or other_schema is None
                or not other_schema.has(other_name)
                or not computed_type.equals(other_schema.field(other_name).type)
            ):
                return None
            # Stored oriented `left_key OP right_key`, so flip when the computation is on
            # the right-hand input.
            if expr_side == "left":
                return _Candidate(None, other_name, op, left_expr=rewritten)
            return _Candidate(other_name, None, ORDERING_FLIP[op], right_expr=rewritten)
    return None
