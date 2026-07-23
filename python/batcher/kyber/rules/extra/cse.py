"""Common-subexpression elimination — compute a repeated expression once, not N times.

A `Project` evaluates each of its output expressions independently over the batch, so a
subexpression that appears in several of them is evaluated once *per appearance*. That is
free to ignore when the repeat is `a + b`, and expensive when it is a regex match, a JSON
extraction, a cast chain, or a decoded image — the shapes a real query repeats most:

    SELECT regexp_extract(url, p) AS host,
           upper(regexp_extract(url, p)) AS host_up,
           length(regexp_extract(url, p)) AS host_len
    FROM t

evaluates `regexp_extract` three times per row. This rule binds it to one synthetic column
and rewrites the outputs to read that column, so the engine evaluates it once:

    Project[host=__bt_cse_0, host_up=upper(__bt_cse_0), host_len=length(__bt_cse_0)]
      Project[url, p, __bt_cse_0=regexp_extract(url, p)]

**Why this is safe.** The bound expression is the *same expression*, evaluated over the
same input rows, so its per-row value is identical by construction — no algebraic identity
is being appealed to, and no type changes (the synthetic column's type is the expression's
own inferred type). The only requirement is that the expression be a pure function of the
row, which is why `_is_hoistable` refuses anything whose value is not determined by the
row alone (see there).

**Why it is cost-gated.** Binding costs one extra materialized column, so hoisting a cheap
subexpression makes the plan *slower* — a vectorized `a + b` over a morsel is cheaper than
allocating a column to hold it. The rule fires only when the saved work,
`(occurrences - 1) x expr_cost`, exceeds `filter_split_materialize_cost` — the same
per-column materialization price, in the same per-row work units, that the filter-splitting
rule pays. `expr_cost` is JIT-aware, so an expression the Cranelift tier compiles is
correctly priced as cheap and is left alone.

Nested repeats share correctly: candidates are bound smallest-first, and a larger
candidate's definition is written in terms of the smaller ones already bound, so
`(a+b)` inside `(a+b)*2` is computed once and reused rather than twice.
"""

from __future__ import annotations

from collections import Counter

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.plan.expr_ir import AggExpr, Col, Expr, Lit, WindowExpr, referenced_columns
from batcher.plan.expr_rewrite import expr_key, replace_subtrees, subexpressions
from batcher.plan.logical import LogicalPlan, Project, Projection

__all__ = ["project_common_subexpression"]

# Prefix of the synthetic column a bound subexpression lands in. Mirrors the existing
# `WINDOW_TEMP_PREFIX` convention: leading dunder-ish underscores make it un-typeable as a
# user column, so a binding can never shadow one the query actually selects.
CSE_TEMP_PREFIX = "__bt_cse_"

# Expression node types whose value is NOT a pure function of the row, or which are not
# scalar in a projection at all. A `Project` should never carry these (windows are hoisted
# into their own `Window` node, aggregates live on `Aggregate`), but a rewrite that binds
# one to a column would change *when* and *over what frame* it is evaluated — so refuse
# them outright rather than rely on an upstream invariant.
_NON_SCALAR = (AggExpr, WindowExpr)


def _is_hoistable(expr: Expr) -> bool:
    """Whether `expr` may be bound to a column without changing what it computes.

    A leaf is pointless to bind: a `Col` is already a column, and a `Lit` is free. Anything
    containing a window or aggregate is refused — binding one would move it out of the frame
    or grouping that defines it. Everything else in the scalar `Expr` algebra is a pure,
    deterministic function of the row, which is exactly the property that makes evaluating
    it once and reusing the result indistinguishable from evaluating it N times.
    """
    if isinstance(expr, Col | Lit):
        return False
    return not any(isinstance(sub, _NON_SCALAR) for sub in subexpressions(expr))


def _size(expr: Expr) -> int:
    """Node count — the ordering that guarantees a candidate is bound after its own parts."""
    return sum(1 for _ in subexpressions(expr))


@rule(name="project_common_subexpression", phase=Phase.FUSION, matches=(Project,))
def project_common_subexpression(node: Project, ctx: OptimizerContext) -> LogicalPlan | None:
    """Bind a `Project`'s repeated subexpressions to columns, computing each once."""
    if len(node.items) < 2:
        return None  # a single output cannot share a subexpression with another

    census: Counter[str] = Counter()
    by_key: dict[str, Expr] = {}
    for item in node.items:
        for sub in subexpressions(item.expr):
            if not _is_hoistable(sub):
                continue
            key = expr_key(sub)
            census[key] += 1
            by_key.setdefault(key, sub)

    cost_of = ctx.costs().expr_cost
    materialize = ctx.config.optimizer.filter_split_materialize_cost
    # Worth binding only when the evaluations saved outweigh the column it costs to hold.
    candidates = [
        by_key[key]
        for key, n in census.items()
        if n >= 2 and (n - 1) * cost_of(by_key[key]) > materialize
    ]
    if not candidates:
        return None

    # Smallest first, so a candidate is numbered after every candidate it contains.
    candidates.sort(key=_size)
    names = {expr_key(c): f"{CSE_TEMP_PREFIX}{i}" for i, c in enumerate(candidates)}

    # A `Project`'s items all evaluate over its *input*, so they cannot reference one
    # another: a binding for `f(f(url))` cannot read the binding for `f(url)` in the same
    # projection. Nested candidates therefore need a **chain** of projections — one per
    # dependency level — which is the let-block this rule is really building. `level` is
    # the depth of a candidate's nesting inside other candidates; everything at one level
    # is independent and shares a single projection.
    level: dict[str, int] = {}
    for cand in candidates:  # ascending size ⇒ contained candidates are already levelled
        inner = [
            level[k]
            for sub in subexpressions(cand)
            if sub is not cand and (k := expr_key(sub)) in names
        ]
        level[expr_key(cand)] = 1 + max(inner) if inner else 0

    definitions: dict[str, Expr] = {}  # column name -> its defining expression
    for cand in candidates:
        key = expr_key(cand)
        definitions[names[key]] = replace_subtrees(cand, _below(level[key], names, level))
    bound = {key: Col(name) for key, name in names.items()}  # expr_key -> the Col holding it

    rewritten = tuple(
        Projection(item.alias, replace_subtrees(item.expr, bound)) for item in node.items
    )
    if all(a.expr is b.expr for a, b in zip(rewritten, node.items, strict=True)):
        return None  # nothing referenced a binding (defensive; the census says otherwise)

    # Every column any surviving expression reads — the input columns the bindings need,
    # plus the bindings themselves. Each projection in the chain carries forward exactly
    # these and nothing else, so the rewrite never re-widens a scan that projection
    # pushdown already pruned.
    needed: set[str] = set()
    for item in rewritten:
        needed |= referenced_columns(item.expr)
    for expr in definitions.values():
        needed |= referenced_columns(expr)

    current: LogicalPlan = node.input
    for lv in range(max(level.values()) + 1):
        carried = [Projection(c, Col(c)) for c in current.available_columns() if c in needed]
        fresh = [
            Projection(names[key], definitions[names[key]])
            for key, depth in level.items()
            if depth == lv
        ]
        current = Project(current, tuple(carried + fresh))
    return Project(current, rewritten)


def _below(mine: int, names: dict[str, str], level: dict[str, int]) -> dict[str, Expr]:
    """The bindings a level-`mine` definition may reference: those at a strictly lower level.

    A candidate cannot read a binding made in its own projection (items evaluate over the
    input, not over each other), and by construction every candidate it contains sits at a
    lower level — so this is exactly the set it may substitute, and it is already emitted by
    the time this level's projection is built.
    """
    return {key: Col(name) for key, name in names.items() if level[key] < mine}
