"""Ordered sargable transposition proved by a column's **measured min/max**.

The half `sargable.py` cannot do. It transposes constant arithmetic across `=`/`<>`
unconditionally and refuses the ordered comparisons, because the engine's i64 arithmetic
wraps: at `INT64_MAX`, `col + 5 > 10` is false while `col > 5` is true. Wrapping breaks
monotonicity, and an ordered comparison needs monotonicity.

It breaks it only at the ends of the range. Where the column's recorded bounds show that
`col ± k` stays inside i64, the arithmetic is exact rather than modular and the ordinary
integer identity holds on every row — so `key - 1 >= 20240101` over a column whose footer
says `[20200101, 20241231]` becomes `key >= 20240102`, and only then does the scan get a
predicate it can push and the zone-map pruner a comparison it can refute. Written the first
way both are blind and the query reads every row group.

The bounds come from wherever `RelStats` has them: a Parquet footer, an ORC index, a
lakehouse manifest, an in-memory table's exact statistics, or Kyber's learned column stats.
`zonemap_prune_filter` already trusts the same numbers to delete whole row groups, so relying
on them for a strictly weaker conclusion — a rewrite rather than an elimination — is
consistent with what the optimizer already does.

**One rule, twelve rewrites.** Three arithmetic forms (`col + k`, `col - k`, `k - col`) times
four comparisons, all in a single traversal. The expression-algebra families split a rewrite
per operator because the driver fuses declared leaves into one shared walk, so the split is
free there. This rule needs the estimator, which no leaf receives, so it is a node rule — and
a node rule pays a full expression traversal *each*. Twelve of them measured +7% on planning
time for a plan carrying the shape, against one that costs a twelfth of that for the same
rewrites.
"""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.kyber.rules.exprs.guards import is_integer, node_schema
from batcher.kyber.rules.extra.sargable_range.shared import ORDERED, decompose, transpose
from batcher.plan.expr_ir import Binary, Expr
from batcher.plan.expr_rewrite import map_node_expressions, transform_expr_up
from batcher.plan.logical import Aggregate, Filter, LogicalPlan, Project, Sort, Window
from batcher.plan.stats import ColumnStat

__all__ = ["sarg_bounded_ordered"]


def _int_bounds(stat: ColumnStat) -> tuple[int, int] | None:
    """The column's recorded integer range, or ``None`` when it has none.

    Both endpoints must be plain integers. A float min/max belongs to a float column, whose
    arithmetic this rule must not touch at all (folding the constant would round differently
    from the engine), and a `bool` is not a value this arithmetic applies to.
    """
    low, high = stat.min, stat.max
    for value in (low, high):
        if not isinstance(value, int) or isinstance(value, bool):
            return None
    return low, high


@rule(
    name="sarg_bounded_ordered",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project, Aggregate, Sort, Window),
    expr_matches=(Binary,),
    # Every ordered comparison, in both spellings: a predicate written `500 < x + 1` arrives
    # as a `lt` and `decompose` normalizes it to `gt`. The set is closed under mirroring, so
    # naming the four covers both directions.
    expr_ops=ORDERED,
)
def sarg_bounded_ordered(node: LogicalPlan, ctx: OptimizerContext) -> LogicalPlan | None:
    """Transpose constant arithmetic out of an ordered comparison, where the column's recorded
    bounds prove the arithmetic cannot wrap: `col + k < lit` becomes `col < lit - k`, exposing
    the raw column that zone-map pruning and source predicate pushdown match on.

    Handles `col + k`, `col - k`, and `k - col` (which also covers unary minus, since `-col`
    lowers to `0 - col` and flips the comparison) against each of `<`, `<=`, `>`, `>=`.
    Declines — leaving the predicate exactly as written — unless the column is provably
    integer-typed, its bounds keep both endpoints of `col ± k` inside i64, and the folded
    literal is itself representable, so no rewrite introduces a value the engine would wrap.
    """
    # Resolved on first candidate, not up front: `node_schema` rebuilds a pyarrow schema up
    # the plan, and almost every node carries nothing this rule can act on.
    resolved: list = []

    def context():
        if not resolved:
            resolved.append((node_schema(node), ctx.estimator.estimate(node.input)))
        return resolved[0]

    def leaf(expr: Expr) -> Expr:
        found = decompose(expr)
        if found is None:
            return expr
        form, op, col, k, lit = found
        schema, stats = context()
        # Excluded by *type*, not merely by its bounds: transposing a constant across a float
        # comparison changes the rounding, whatever the column's range is.
        if not is_integer(col, schema):
            return expr
        bounds = _int_bounds(stats.column(col.name))
        if bounds is None:
            return expr
        rewritten = transpose(form, op, col, k, lit, *bounds)
        return expr if rewritten is None else rewritten

    rebuilt = map_node_expressions(node, lambda e: transform_expr_up(e, leaf))
    return None if rebuilt is node else rebuilt
