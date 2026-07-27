"""Boolean-predicate normalizations in the NORMALIZE phase.

Four rewrites the reference optimizers carry and this one did not. Each takes a
construct that is *correct but opaque* and gives it a shape the rest of Kyber can
already act on, so the value is not the saved node — it is every downstream rule that
was blind to the original spelling.

* `self_comparison_to_null_check` (DuckDB `comparison_simplification`) — `a = a` is
  `a IS NOT NULL`, which zone-map pruning and null-count metadata can answer without
  reading a row.
* `boolean_case_to_predicate` (DuckDB `case_simplification`) — `CASE WHEN c THEN true
  ELSE false END` is the branch condition itself. The swapped form is not, and the
  docstring says why.
* `intersect_in_lists` (DuckDB `in_clause_simplification`) — two `IN` lists on one
  column under an `AND` become their intersection, which is what bloom probing and
  source pushdown actually send down.
* `constant_group_key_removed` (DataFusion `eliminate_group_by_constant`) — a literal
  grouping key builds a one-entry hash table and forces a single-target shuffle; the
  key comes back as a projection so the schema is unchanged.

The first three are null-sensitive, and the three-valued detail is the whole reason they
are written out rather than lifted from the reference verbatim: each fires under a
`Filter` and nowhere else, because that is the one context where NULL and FALSE are
indistinguishable. `boolean_case_to_predicate` is narrower still and fires in only one
of its two directions. See the individual docstrings.
"""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import DEFAULT_REGISTRY, rule
from batcher.kyber.rule import Phase, plan_rule
from batcher.plan.expr_ir import Binary, Case, Col, Expr, InList, IsNotNull, Lit
from batcher.plan.expr_rewrite import expr_key
from batcher.plan.logical import Aggregate, Filter, LogicalPlan, Project, Projection
from batcher.plan.visitor import transform_up

__all__ = [
    "boolean_case_to_predicate",
    "constant_group_key_removed",
    "intersect_in_lists",
    "self_comparison_to_null_check",
]


# --- `a = a` → `a IS NOT NULL` ----------------------------------------------

# Comparisons that hold for every non-null value, so the whole predicate reduces to
# "is this row's value present". `ne`/`lt`/`gt` are the complement and are *not* here:
# they are FALSE when non-null and NULL when null, which is not `Lit(False)` in a
# projection, so folding them needs a context this rule does not have.
_REFLEXIVE = frozenset({"eq", "le", "ge"})


def self_comparison_to_null_check(plan: LogicalPlan) -> LogicalPlan:
    """Rewrite `a = a` (and `a <= a`, `a >= a`) under a filter to `a IS NOT NULL`.

    The comparison is true for every value a column can hold, so the only information
    left in it is whether the value is null. Stated as `IS NOT NULL`, the predicate can
    be answered from a column's null count in the metadata Kyber already carries, and a
    row group with no nulls can be skipped without being read; as a self-comparison it
    forces a full scan and a per-row evaluation.

    **The two forms are not interchangeable as values.** `a = a` on a null row is NULL,
    while `a IS NOT NULL` is FALSE — a difference a filter erases (both drop the row)
    and a projection does not. So this fires on a `Filter` predicate only, at any depth
    inside it, since every operator that can appear between two conjuncts of a filter
    treats NULL and FALSE alike.

    It fires only for a bare `Col` on both sides. A general expression would need to be
    proven deterministic and side-effect-free first, and Batcher has no such analysis;
    a column reference needs none.
    """
    return transform_up(plan, _rewrite_self_comparison_in_filter)


def _rewrite_self_comparison_in_filter(node: LogicalPlan) -> LogicalPlan:
    if not isinstance(node, Filter):
        return node
    return Filter(node.input, _rewrite_self_comparison(node.predicate))


def _rewrite_self_comparison(expr: Expr) -> Expr:
    if (
        isinstance(expr, Binary)
        and expr.op in _REFLEXIVE
        and isinstance(expr.left, Col)
        and isinstance(expr.right, Col)
        and expr.left.name == expr.right.name
    ):
        return IsNotNull(expr.left)
    if isinstance(expr, Binary) and expr.op in ("and", "or"):
        return Binary(
            expr.op, _rewrite_self_comparison(expr.left), _rewrite_self_comparison(expr.right)
        )
    return expr


DEFAULT_REGISTRY.add(
    plan_rule(
        "self_comparison_to_null_check",
        Phase.NORMALIZE,
        lambda plan, _ctx: self_comparison_to_null_check(plan),
    )
)


# --- `CASE WHEN c THEN true ELSE false END` → `c` ---------------------------


def boolean_case_to_predicate(plan: LogicalPlan) -> LogicalPlan:
    """Rewrite `CASE WHEN c THEN true ELSE false END` under a filter to `c`.

    A `CASE` is a barrier: it is not a comparison, so no range is derived from it, it is
    not pushed into a source, and it cannot become a join key — while the condition
    inside it is often all three.

    **The null case decides how far this can go, and it is asymmetric.** A `CASE` sends
    a NULL condition down the `ELSE` branch, so the whole expression is `c IS TRUE`:

    * `THEN true ELSE false` is `c IS TRUE`, which *under a filter* keeps exactly the
      rows `c` keeps — NULL and FALSE both drop — so the rewrite to `c` is sound there
      and nowhere else;
    * `THEN false ELSE true` is `c IS NOT TRUE`, which **keeps** a NULL row, while
      `NOT c` is NULL and drops it. That form is therefore left alone. It is not a
      missing case; it is a rewrite that would be wrong, and the differential fixture
      carries a null specifically to hold that line.

    So the rule fires only on a `Filter` predicate, only at its top, and only in the
    one direction. A `CASE` nested inside a larger expression may be feeding something
    that distinguishes NULL from FALSE, such as a `COALESCE` or an `IS NULL`.
    """
    return transform_up(plan, _rewrite_boolean_case_in_filter)


def _rewrite_boolean_case_in_filter(node: LogicalPlan) -> LogicalPlan:
    if not isinstance(node, Filter):
        return node
    rewritten = _boolean_case_condition(node.predicate)
    return node if rewritten is None else Filter(node.input, rewritten)


def _boolean_case_condition(expr: Expr) -> Expr | None:
    """The branch condition of a `THEN true ELSE false` `CASE`, or None.

    The swapped form returns None on purpose — see the rule docstring: `NOT c` drops the
    null rows that `CASE ... THEN false ELSE true END` keeps.
    """
    if not (isinstance(expr, Case) and len(expr.branches) == 1):
        return None
    (condition, then), otherwise = expr.branches[0], expr.otherwise
    if not (_is_bool_lit(then) and _is_bool_lit(otherwise)):
        return None
    if not (then.value and not otherwise.value):
        return None
    return condition


def _is_bool_lit(expr: Expr) -> bool:
    # `type(...) is bool` because `Lit(1)` is not the boolean literal this matches, and
    # `1 == True` in Python.
    return isinstance(expr, Lit) and type(expr.value) is bool


DEFAULT_REGISTRY.add(
    plan_rule(
        "boolean_case_to_predicate",
        Phase.NORMALIZE,
        lambda plan, _ctx: boolean_case_to_predicate(plan),
    )
)


# --- `x IN (1,2,3) AND x IN (2,3,4)` → `x IN (2,3)` -------------------------


@rule(name="intersect_in_lists", phase=Phase.NORMALIZE, matches=(Filter,))
def intersect_in_lists(node: Filter, _ctx: OptimizerContext) -> LogicalPlan:
    """Intersect two `IN` lists on the same column joined by `AND` (DuckDB's
    `in_clause_simplification`).

    Two `IN` lists on one column are a redundant pair: the row must be in both, so it
    must be in the intersection, and every value outside it is probed for nothing. The
    shape is common because it is what a view predicate ANDed with a user predicate
    looks like after `or_equalities_to_in_list` has run on each half independently —
    neither rule can see the other's output, so the intersection is left on the table.

    Shrinking the list is worth more than the saved comparisons: the list is what bloom
    probing and source pushdown send down, and a narrower one prunes more row groups.

    Nullness is preserved because the intersection is non-empty by construction — the
    rule declines the empty case. `x IN (1) AND x IN (2)` is FALSE for a non-null `x`
    but NULL for a null one, so folding it to `Lit(False)` would be right under this
    filter and wrong if the predicate is ever read as a value; the rewrite that is only
    conditionally sound is the one not made.
    """
    rewritten = _intersect_conjoined_in_lists(node.predicate)
    return node if rewritten is None else Filter(node.input, rewritten)


def _intersect_conjoined_in_lists(expr: Expr) -> Expr | None:
    if not (isinstance(expr, Binary) and expr.op == "and"):
        return None
    left, right = expr.left, expr.right
    if not (isinstance(left, InList) and isinstance(right, InList)):
        return None
    if expr_key(left.input) != expr_key(right.input):
        return None
    # Order follows the left list so the rewrite is deterministic; `values` may hold
    # unhashable literals, so membership is a scan rather than a set intersection.
    kept = tuple(v for v in left.values if v in right.values)
    if not kept or len(kept) == len(left.values):
        return None  # empty (see the docstring) or already the intersection
    return InList(left.input, kept)


# --- `GROUP BY <constant>` → group by nothing -------------------------------


@rule(name="constant_group_key_removed", phase=Phase.NORMALIZE, matches=(Aggregate,))
def constant_group_key_removed(node: Aggregate, _ctx: OptimizerContext) -> LogicalPlan:
    """Drop a constant grouping key (DataFusion's `eliminate_group_by_constant`).

    A literal key puts every row in one bucket, so the hash table it builds has exactly
    one entry and every row pays a hash, a probe and a key comparison to reach it. With
    the key gone the operator is a plain global aggregate, which streams: no hash table,
    no key columns carried through the partial state, and — the part that matters at
    scale — no repartition, since a single-group shuffle sends the whole relation to one
    node while a keyless aggregate merges partials from all of them.

    The key still has to appear in the output, so it comes back as a projected literal
    above the aggregate. That keeps the schema, the column order and the name identical;
    a rewrite that quietly dropped a column would be a wrong answer, not a faster one.

    All-constant keys are left alone: `GROUP BY 1` over an empty input produces no rows,
    while a global aggregate over an empty input produces one, and that difference is
    visible. Only the mixed case — at least one real key beside the constant — is
    rewritten, where the row count is decided by the real keys either way.

    The constant is usually not written at the key: `with_columns(k=lit(1)).group_by(k)`
    puts the literal in a projection and leaves a plain `Col("k")` on the key, so the
    rule resolves one level through an immediately-underlying `Project`. The now-unused
    projection item is left for projection pruning rather than removed here, which keeps
    this rule to one concern.
    """
    constants = _constant_group_keys(node)
    if not constants or len(constants) == len(node.group_keys):
        return node
    kept = [k for k in node.group_keys if k.alias not in constants]
    grouped = Aggregate(node.input, kept, node.aggregates, node.watermark)
    # Rebuild the original output order: keys in their original positions, then aggregates.
    # `constants.get(...) or Col(...)` would ask an `Expr` for its truth value, which
    # `Expr.__bool__` rejects on purpose.
    items = [
        Projection(k.alias, constants[k.alias] if k.alias in constants else Col(k.alias))
        for k in node.group_keys
    ] + [Projection(a.alias, Col(a.alias)) for a in node.aggregates]
    return Project(grouped, tuple(items))


def _constant_group_keys(node: Aggregate) -> dict[str, Lit]:
    """Group-key alias → the literal it is, for every key with a constant value."""
    below: dict[str, Lit] = {}
    if isinstance(node.input, Project):
        below = {i.alias: i.expr for i in node.input.items if isinstance(i.expr, Lit)}
    constants: dict[str, Lit] = {}
    for key in node.group_keys:
        if isinstance(key.expr, Lit):
            constants[key.alias] = key.expr
        elif isinstance(key.expr, Col) and key.expr.name in below:
            constants[key.alias] = below[key.expr.name]
    return constants
