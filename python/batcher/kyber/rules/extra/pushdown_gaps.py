"""Pushdown gaps — the operators a `Filter`/projection may legally descend past, but didn't.

The base pushdown rules sink a `Filter` through `Project`, `Sort`, `Window`, `Distinct`,
`Aggregate`, `Union` and `Join`, and the whole-plan `projection_rewrite` prunes columns
down to the scans. That leaves the *reshaping* operators — `Unnest`, `Unpivot`, `Sample`,
`RowId`, `AsofJoin` — with no local pushdown at all, so a predicate over columns those
operators merely carry through is evaluated on the (much larger) reshaped relation. This
module closes exactly those gaps. Every rule descends only past an operator that neither
invents rows nor rewrites the values the predicate reads.

What is deliberately **refused** (a wrong answer, not a slow one):

- **`Filter` below `RowId`.** `with_row_index` numbers rows by *position*, so removing rows
  first renumbers them: `Filter(RowId(x), p)` keeps the original positions of the surviving
  rows, `RowId(Filter(x, p))` renumbers them `0..k`. Different values. The only sound move is
  to delete a row-index nothing reads — `drop_dead_row_index`.
- **`Filter` below a fixed-count `Sample(n=…)`.** It keeps the `n` smallest-hash rows of the
  whole input; filtering first changes *which* `n` rows win. Only the `fraction` mode — a pure
  per-row content hash — commutes (`push_filter_through_sample`).
- **`Filter` on an ASOF join's *right* columns.** ASOF matches the nearest right row and only
  then would the predicate see it; pre-filtering the right side lets a *farther* right row
  become the nearest match. Only left-column and `by`-key predicates descend.
- **Column pruning below `Sample` / a `distinct` `Union`.** Both hash the whole row, so
  dropping a column changes which rows survive.
"""

from __future__ import annotations

import dataclasses

import pyarrow as pa

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.plan.expr_ir import Binary, Col, Expr, ListContains, Lit, referenced_columns
from batcher.plan.expr_rewrite import combine_conjuncts, split_conjuncts, substitute_columns
from batcher.plan.logical import (
    Aggregate,
    AsofJoin,
    Filter,
    Join,
    LogicalPlan,
    Project,
    Projection,
    RowId,
    Sample,
    Union,
    Unnest,
    Unpivot,
)

__all__ = [
    "drop_dead_row_index",
    "prefilter_unnest_by_list_contains",
    "prune_asof_join_output",
    "prune_union_columns_under_aggregate",
    "prune_union_columns_under_join",
    "prune_union_columns_under_unpivot",
    "push_filter_into_asof_by_keys",
    "push_filter_into_unpivot_columns",
    "push_filter_through_asof_join",
    "push_filter_through_sample",
    "push_filter_through_unnest",
    "push_filter_through_unpivot",
]


def _split_pushable(predicate: Expr, allowed: set[str]) -> tuple[list[Expr], list[Expr]]:
    """Split `predicate`'s conjuncts into (those reading only `allowed`, the rest)."""
    pushable: list[Expr] = []
    keep: list[Expr] = []
    for conj in split_conjuncts(predicate):
        (pushable if referenced_columns(conj) <= allowed else keep).append(conj)
    return pushable, keep


def _agg_child_columns(node: Aggregate) -> set[str]:
    """Every input column an `Aggregate` reads: its group keys and its aggregate arguments."""
    need: set[str] = set()
    for key in node.group_keys:
        need |= referenced_columns(key.expr)
    for spec in node.aggregates:
        if spec.agg.input is not None:
            need |= referenced_columns(spec.agg.input)
        if spec.agg.input2 is not None:  # arg_min/arg_max carry an ordering key
            need |= referenced_columns(spec.agg.input2)
    return need


def _prune_union(union: LogicalPlan, keep: set[str]) -> LogicalPlan | None:
    """A copy of `union` whose branches produce only `keep`, or None if nothing prunes.

    Every branch gets the *same* column list, in the union's own column order, so the
    branches stay schema-identical (a `Union` requires that) and the union's output order is
    a subsequence of the original — no consumer sees a reordered or renamed column.

    Refused for a **distinct** union: `UNION` deduplicates over every column, so dropping one
    merges rows that were distinct. A branch must also keep at least one column (a relation
    with no columns has no rows to count), so an empty requirement falls back to the first.
    """
    if not isinstance(union, Union) or union.distinct:
        return None
    cols = union.available_columns()
    kept = [c for c in cols if c in keep] or cols[:1]
    if len(kept) == len(cols):
        return None
    items = tuple(Projection(c, Col(c)) for c in kept)
    return Union(tuple(Project(i, items) for i in union.inputs), distinct=False)


@rule(name="push_filter_through_unnest", phase=Phase.PUSHDOWN, matches=(Filter,))
def push_filter_through_unnest(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Filter(Unnest(x, c, a), p)` → `Unnest(Filter(x, p), c, a)` when `p` reads only
    columns `Unnest` carries through unchanged.

    `Unnest` replaces the list column `c` with its elements bound to `a` and repeats every
    other column once per element. A pass-through column therefore has the *same value* on
    every output row a given input row produces, so `p` accepts either all of that input
    row's output rows or none of them — exactly the rows `Filter(x, p)` keeps. Filtering
    first means the explode never materializes the rows that were going to be dropped.

    Refuses any conjunct touching the exploded output `a` (its value is per-element and does
    not exist below the `Unnest`), and refuses outright if the alias shadows a pass-through
    column (the name would be ambiguous). Mixed predicates push their pass-through conjuncts
    and keep the rest above.
    """
    unnest = node.input
    if not isinstance(unnest, Unnest):
        return None
    passthrough = set(unnest.input.available_columns()) - {unnest.column}
    if unnest.alias in passthrough:
        return None  # the exploded alias shadows a carried column — names are ambiguous
    pushable, keep = _split_pushable(node.predicate, passthrough)
    if not pushable:
        return None
    pushed = Unnest(Filter(unnest.input, combine_conjuncts(pushable)), unnest.column, unnest.alias)
    return pushed if not keep else Filter(pushed, combine_conjuncts(keep))


@rule(name="prefilter_unnest_by_list_contains", phase=Phase.PUSHDOWN, matches=(Filter,))
def prefilter_unnest_by_list_contains(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Filter(Unnest(x, c, a), a = v ∧ …)` → the same, over `Filter(x, list_contains(c, v))`.

    The one sound way a predicate on the *exploded* column reaches below the explode. A row
    of `x` whose list does not contain `v` produces only output rows with `a ≠ v`, every one
    of which `a = v` discards — so dropping that row before the explode cannot lose an output
    row. (A null or empty list already explodes to nothing, and `list_contains` is null/false
    there, so those rows are dropped identically.) The pre-filter is a *superset* selection,
    which is why the original `Filter` is kept above it: a surviving row's list may hold other
    elements too, and those output rows must still be discarded.

    Guarded to a plain `List` column with a non-float element type and a non-float literal, so
    `list_contains` and the post-explode `a = v` run the same equality on the same types and
    cannot disagree (a float `-0.0`/`NaN` mismatch would drop a row that should survive).
    Returns None when the pre-filter is already present, so the rule is idempotent.
    """
    unnest = node.input
    if not isinstance(unnest, Unnest):
        return None
    value = _eq_literal(node.predicate, unnest.alias)
    if value is None or isinstance(value, float):
        return None
    schema = unnest.input.available_schema()
    if schema is None:
        return None
    list_type = schema.field(unnest.column).type
    if not pa.types.is_list(list_type) or pa.types.is_floating(list_type.value_type):
        return None
    guard = ListContains(Col(unnest.column), value)
    if guard.to_ir() in _conjunct_irs(unnest.input):
        return None  # already pre-filtered — the rule has run (idempotence)
    below = [*_conjuncts_of(unnest.input), guard]
    source = unnest.input.input if isinstance(unnest.input, Filter) else unnest.input
    filtered = Unnest(Filter(source, combine_conjuncts(below)), unnest.column, unnest.alias)
    return Filter(filtered, node.predicate)


def _eq_literal(predicate: Expr, column: str) -> int | float | bool | str | None:
    """The literal `v` of a `column = v` conjunct of `predicate` (either operand order)."""
    for conj in split_conjuncts(predicate):
        if not isinstance(conj, Binary) or conj.op != "eq":
            continue
        for lhs, rhs in ((conj.left, conj.right), (conj.right, conj.left)):
            if isinstance(lhs, Col) and lhs.name == column and isinstance(rhs, Lit):
                return rhs.value
    return None


@rule(name="push_filter_through_unpivot", phase=Phase.PUSHDOWN, matches=(Filter,))
def push_filter_through_unpivot(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Filter(Unpivot(x, index, on, …), p)` → `Unpivot(Filter(x, p), …)` when `p` reads only
    `index` columns.

    An unpivot turns each input row into one output row per melted column, repeating the
    `index` (identifier) columns verbatim on each. An `index`-only predicate therefore holds
    for all of a row's output rows or none, so filtering the input first drops exactly the
    same output rows — before the reshape multiplies them by `len(on)`.

    Refuses a predicate touching the synthesized `variable`/`value` columns (they exist only
    above the unpivot; a `variable` predicate has its own rule), and refuses outright if
    either synthesized name shadows an `index` column. Mixed predicates split.
    """
    unpivot = node.input
    if not isinstance(unpivot, Unpivot):
        return None
    index = set(unpivot.index)
    if unpivot.variable_name in index or unpivot.value_name in index:
        return None  # a synthesized name shadows an identifier column — ambiguous
    pushable, keep = _split_pushable(node.predicate, index)
    if not pushable:
        return None
    pushed = dataclasses.replace(unpivot, input=Filter(unpivot.input, combine_conjuncts(pushable)))
    return pushed if not keep else Filter(pushed, combine_conjuncts(keep))


@rule(name="push_filter_into_unpivot_columns", phase=Phase.PUSHDOWN, matches=(Filter,))
def push_filter_into_unpivot_columns(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Filter(Unpivot(x, on=[a, b, c], …), variable = 'b')` → melt only `b`.

    The `variable` column holds the *name* of the melted column each output row came from, so
    a `variable = 'b'` predicate keeps precisely the rows `on = ('b',)` would have produced —
    and in the same order (the unpivot emits its melted columns in `on` order, so restricting
    `on` deletes whole blocks and never reorders what remains). The other melted columns are
    then never read at all.

    The output *schema* must not move: the `value` column's type is the promotion of every
    `on` column's type, so this fires only when they already share one type (then any subset
    promotes to the same type). The `Filter` is kept above — it is now always-true, but
    dropping it is `prune_true_filter`'s job, and keeping it makes this rule idempotent
    (`on` no longer changes on a second pass).
    """
    unpivot = node.input
    if not isinstance(unpivot, Unpivot) or len(unpivot.on) < 2:
        return None
    wanted = _eq_literal(node.predicate, unpivot.variable_name)
    if not isinstance(wanted, str):
        return None
    kept = tuple(c for c in unpivot.on if c == wanted)
    if not kept or len(kept) == len(unpivot.on):
        return None
    schema = unpivot.input.available_schema()
    if schema is None or len({schema.field(c).type for c in unpivot.on}) != 1:
        return None  # a mixed-type melt: narrowing `on` would change the value column's type
    return Filter(dataclasses.replace(unpivot, on=kept), node.predicate)


@rule(name="push_filter_through_sample", phase=Phase.PUSHDOWN, matches=(Filter,))
def push_filter_through_sample(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Filter(Sample(x, fraction, seed), p)` → `Sample(Filter(x, p), fraction, seed)`.

    A **fraction** sample is a pure per-row predicate: the engine keeps a row iff a seeded
    hash of that row's own encoded values falls under the threshold (`ops::reshape::
    sample_batch`) — it depends on no other row, on no batch boundary, and on no worker
    count. A filter changes which rows are present but never a row's values, so the two
    predicates commute: both orders keep `{r ∈ x : p(r) ∧ hash(r) ≤ threshold}`. Filtering
    first means the sampler row-encodes and hashes far fewer rows.

    Refused for the fixed-count mode (`n is not None`): that keeps the `n` globally smallest
    hashes, so removing rows first promotes rows that would have lost, and the sampled set
    changes. (For the same reason, nothing may prune a *column* below a sample — the hash
    covers the whole row.)
    """
    sample = node.input
    if not isinstance(sample, Sample) or sample.n is not None:
        return None
    return dataclasses.replace(sample, input=Filter(sample.input, node.predicate))


@rule(name="push_filter_through_asof_join", phase=Phase.PUSHDOWN, matches=(Filter,))
def push_filter_through_asof_join(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Filter(AsofJoin(l, r), p)` → `AsofJoin(Filter(l, p'), r)` when `p` reads only
    left-side output columns.

    An ASOF join is left-style: it emits **exactly one output row per left row** (null right
    columns when nothing matches), and each left row's match depends only on that row's `on`
    and `by` values — never on the other left rows. So dropping left rows before the join
    removes precisely the output rows the filter would have removed, and leaves every
    surviving row's match untouched. `p'` is `p` rewritten from output aliases to the left
    input's own column names.

    Refuses any conjunct reading a **right** column, and this is the load-bearing refusal:
    the join picks the *nearest* right row and the predicate only sees it afterwards, so
    pre-filtering the right side can promote a farther right row to nearest — a different
    answer, not just a different plan. Mixed predicates push their left-only conjuncts and
    keep the rest above.
    """
    asof = node.input
    if not isinstance(asof, AsofJoin):
        return None
    left_map = {o.alias: Col(o.name) for o in asof.output if o.side == "left"}
    pushable, keep = _split_pushable(node.predicate, set(left_map))
    if not pushable:
        return None
    pushed_pred = substitute_columns(combine_conjuncts(pushable), left_map)
    pushed = dataclasses.replace(asof, left=Filter(asof.left, pushed_pred))
    return pushed if not keep else Filter(pushed, combine_conjuncts(keep))


@rule(name="push_filter_into_asof_by_keys", phase=Phase.PUSHDOWN, matches=(AsofJoin,))
def push_filter_into_asof_by_keys(node: AsofJoin, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Mirror a left-side `by`-key predicate onto an ASOF join's right input.

    `AsofJoin(Filter(l, p), r)` → `AsofJoin(Filter(l, p), Filter(r, p'))` when `p` reads only
    `by` (exact-match group) columns. A right row can only ever match a left row in the *same*
    `by` group, and `p` is a function of the group alone — so a right row whose group fails `p`
    is unreachable from every left row that survives `p`, and deleting it cannot change any
    match (`by` groups are independent: removing rows from one never re-points a nearest-match
    in another). That shrinks the ASOF build side, the expensive half.

    `p'` is `p` with each left `by` column renamed to its right counterpart. Restricted to
    conjuncts reading `by` columns only — a predicate on any other left column says nothing
    about the right side. Skips conjuncts the right input already carries, so the rule reaches
    a fixpoint instead of stacking a filter every pass.
    """
    if not node.left_by or not isinstance(node.left, Filter):
        return None
    by_map = dict(zip(node.left_by, node.right_by, strict=True))
    if len(by_map) != len(node.left_by):
        return None  # a repeated left `by` key maps ambiguously — refuse
    rename: dict[str, Expr] = {old: Col(new) for old, new in by_map.items()}
    right_cols = set(node.right.available_columns())
    existing = _conjunct_irs(node.right)
    mirrored: list[Expr] = []
    for conj in split_conjuncts(node.left.predicate):
        if not referenced_columns(conj) <= set(by_map):
            continue  # not a pure `by`-group predicate — it says nothing about the right
        remapped = substitute_columns(conj, rename)
        if referenced_columns(remapped) <= right_cols and remapped.to_ir() not in existing:
            mirrored.append(remapped)
    if not mirrored:
        return None
    combined = _conjuncts_of(node.right) + mirrored
    right_input = node.right.input if isinstance(node.right, Filter) else node.right
    return dataclasses.replace(node, right=Filter(right_input, combine_conjuncts(combined)))


def _conjuncts_of(node: LogicalPlan) -> list[Expr]:
    """The top-level conjuncts of a `Filter` node, or `[]` for anything else."""
    return split_conjuncts(node.predicate) if isinstance(node, Filter) else []


def _conjunct_irs(node: LogicalPlan) -> list[dict]:
    """The lowered IR of `node`'s conjuncts — compared by IR, never by `==` (which
    `Expr.__eq__` overloads into building a comparison expression)."""
    return [c.to_ir() for c in _conjuncts_of(node)]


@rule(name="prune_asof_join_output", phase=Phase.PUSHDOWN, matches=(Project,))
def prune_asof_join_output(node: Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Project(AsofJoin(…, output), items)` → drop the `output` columns `items` never read.

    An ASOF join, like an equi-join, *materializes* a fixed output column list — a column no
    consumer reads is still gathered from the matched right rows and carried through the
    join. The whole-plan column pruner narrows an equi-`Join`'s output this way but leaves an
    `AsofJoin`'s intact; this is that missing arm, in the shape where the requirement is
    locally provable: the `Project` directly above declares exactly what it consumes.

    The join's own `on`/`by` keys are read from its *inputs*, not from `output`, so pruning
    an output column never starves the match. The projection's schema is untouched (only
    columns nothing references disappear), and at least one output column is kept so the join
    still has rows. Returns None when nothing is prunable, keeping the rule idempotent.
    """
    asof = node.input
    if not isinstance(asof, AsofJoin):
        return None
    need: set[str] = set()
    for item in node.items:
        need |= referenced_columns(item.expr)
    kept = tuple(o for o in asof.output if o.alias in need) or asof.output[:1]
    if len(kept) == len(asof.output):
        return None
    return Project(dataclasses.replace(asof, output=kept), node.items)


@rule(name="drop_dead_row_index", phase=Phase.PUSHDOWN, matches=(Project, Aggregate))
def drop_dead_row_index(node: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Delete a `RowId` whose index column the consumer directly above never reads.

    `with_row_index` is otherwise a pure pass-through — same rows, same order, same values,
    one extra column — so when the `Project` or `Aggregate` above it reads none of that
    column, the operator computes a counter nobody looks at, and removing it leaves the
    result identical. It also unblocks the pushdowns beneath: a `Filter` may not descend past
    a live `RowId` (numbering by position, it renumbers the survivors), but once the dead
    `RowId` is gone the filter (and column pruning) can reach the scan.

    Only these two consumers, because each fully determines what it reads from its child: a
    `Project` reads its items' columns, an `Aggregate` its keys' and arguments'. A rule firing
    on the `RowId` itself could not know what its *ancestors* read.
    """
    row_id = node.input
    if not isinstance(row_id, RowId):
        return None
    if isinstance(node, Project):
        read = set()
        for item in node.items:
            read |= referenced_columns(item.expr)
    else:
        read = _agg_child_columns(node)
    if row_id.alias in read:
        return None
    return dataclasses.replace(node, input=row_id.input)


@rule(name="prune_union_columns_under_aggregate", phase=Phase.PUSHDOWN, matches=(Aggregate,))
def prune_union_columns_under_aggregate(
    node: Aggregate, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`Aggregate(Union(a, b, …))` → project each branch down to the columns the aggregate reads.

    The whole-plan column pruner hands a `Union` its *full* column list rather than the
    downstream requirement, so a `GROUP BY` over a `UNION ALL` of scans reads every column of
    every branch even when it touches one. An aggregate's requirement is self-contained — its
    group keys and aggregate arguments, and nothing else — which is what makes the pruning
    locally provable here. Each branch gets the same narrowed projection, so they stay
    schema-identical, and the aggregate's expressions still resolve by name.

    Refused for a `distinct` union (see `_prune_union`: `UNION` dedups on every column).
    """
    pruned = _prune_union(node.input, _agg_child_columns(node))
    return None if pruned is None else dataclasses.replace(node, input=pruned)


@rule(name="prune_union_columns_under_join", phase=Phase.PUSHDOWN, matches=(Join,))
def prune_union_columns_under_join(node: Join, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Join(Union(…), …)` → project each union branch down to the columns the join reads.

    The same gap as `prune_union_columns_under_aggregate`, on the other self-contained
    consumer: a join reads exactly its keys plus the source columns named by its `output`
    spec, on each side independently. Narrowing a `UNION ALL` branch to that set shrinks the
    rows that get hashed, built, and probed — and the keys and output columns are kept by
    construction, so the join still validates and its schema is unchanged. Both sides are
    considered; either, or both, may be a union.
    """
    left_need = set(node.left_keys) | {o.name for o in node.output if o.side == "left"}
    right_need = set(node.right_keys) | {o.name for o in node.output if o.side == "right"}
    left = _prune_union(node.left, left_need)
    right = _prune_union(node.right, right_need)
    if left is None and right is None:
        return None
    return dataclasses.replace(
        node,
        left=node.left if left is None else left,
        right=node.right if right is None else right,
    )


@rule(name="prune_union_columns_under_unpivot", phase=Phase.PUSHDOWN, matches=(Unpivot,))
def prune_union_columns_under_unpivot(node: Unpivot, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Unpivot(Union(…))` → project each union branch down to the `index` + `on` columns.

    The third self-contained consumer: an unpivot reads its identifier columns and its melted
    columns, and synthesizes the rest — so a `UNION ALL` branch need produce nothing else.
    The melted `on` columns are all kept, so the `value` column's promoted type is unchanged,
    and the output schema (`index` + `variable` + `value`) does not move.
    """
    pruned = _prune_union(node.input, set(node.index) | set(node.on))
    return None if pruned is None else dataclasses.replace(node, input=pruned)
