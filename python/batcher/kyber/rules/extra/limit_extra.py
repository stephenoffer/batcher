"""LIMIT / top-N rewrites that the existing limit rules leave on the table.

The positional-prefix rules are already covered elsewhere: `combine_limits` collapses
stacked limits, `push_limit_through_project` / `push_limit_through_row_index` /
`push_limit_into_union` / `push_offset_limit_into_union` sink a limit, `drop_inert_limit`
and `empty_limit_past_offset` decide one from an EXACT row count, and `topn_fusion` turns
`Limit(Sort(x))` into a top-N (`Sort.limit`). This module works on what those leave:

* the **fused top-N** itself. `Sort.limit` is set in FUSION, *after* every pushdown rule
  has run, so the sort it lives on is never revisited: `topn_through_project` sinks it
  below a projection (which then evaluates only `limit` rows, not the whole relation),
  `push_topn_into_union` gives each `UNION ALL` branch its own top-N (the classic
  distributed top-N: each branch keeps `k`, the union re-ranks), and
  `collapse_topn_over_topn` merges two same-key top-Ns into one.
* the **row-capping operators the EXACT-stats rules cannot see through**. A fixed-count
  `Sample(n=k)` yields at most `k` rows *structurally*, but the estimator tags it
  `DEFAULT` (it is a bound, not a count), so `drop_inert_limit` never fires on it;
  `drop_limit_over_bounded_sample` and `empty_limit_past_bounded_sample` read the bound
  off the plan instead.
* the **empty marker**. `Limit(x, 0)` is the canonical empty relation, and
  `prune_input_of_empty_limit` deletes the (schema-preserving) work underneath one.
  `empty_topn_to_empty` restores the marker when `topn_fusion` has swallowed it into a
  `Sort(limit=0)`, so `propagate_empty_relation` can still fold it away.
* `prune_sort_keys_after_unique_key` — sort keys *after* a provably-unique one never
  break a tie, so they cannot affect the ordering and are dropped.

Deliberately **not** implemented — a `LIMIT` is a *positional prefix*, so almost nothing
may cross it, and each of these looks plausible and is wrong:

* `push_limit_through_filter` — **unsound**. `Limit(Filter(p, x), n)` takes the first `n`
  rows *that pass* `p`; `Filter(p, Limit(x, n))` takes the passing rows of the first `n`.
  The filter drops rows the limit would have counted, so the second returns *fewer* rows
  (zero, if the first `n` all fail `p`). The same argument rules out pushing a limit
  below `Unnest`/`Unpivot` (a row can expand to zero rows, and the engine's unpivot emits
  column-major within a morsel, so an output prefix is not an input prefix), below
  `Sample` (which keeps a hash-selected subset, not a prefix), below `Distinct`, below an
  `Aggregate`/`Window` (both read the whole partition), and below `MapBatches` (opaque
  row count).
* `push_limit_into_scan` — **not a rule, and no longer missing**. It is not a plan
  rewrite at all: the cap does not change the tree, it rides the physical plan's source
  hand-off beside the projection and the predicate. `PhysicalPlan.source_limits` is that
  channel and `kyber.rules.source_limits.required_limits_per_source` is the analysis that
  fills it, so nothing here needs to move a `Limit`. (This entry used to say the change
  was blocked on a two-sided IR change across the FFI. It was not: that hand-off is a
  Python-side field on `PhysicalPlan`, and `RelOp::Scan` never had to learn anything.)
* `limit_zero_to_empty` / `limit_of_limit` / `drop_limit_larger_than_provable_row_count` /
  `offset_zero_elimination` — already covered: `Limit(x, 0)` *is* the canonical empty
  relation (the estimator even tags it EXACT-0), `combine_limits` merges stacked limits,
  `drop_inert_limit` / `drop_redundant_limit` drop a limit an EXACT row count proves
  inert, and an `offset` of 0 is the `Limit` default — there is no operator to eliminate.
* `push_limit_through_distinct_when_input_is_unique` — subsumed. The only proof of
  uniqueness available is an EXACT ndv reaching the EXACT row count, which is exactly
  `drop_distinct_when_unique`'s condition — and it runs first (REWRITE), deleting the
  `Distinct` outright.
* reordering `Limit` and `Sample` in either direction — **unsound**, and not symmetric
  either: `sample(limit(x))` samples a prefix; `limit(sample(x))` prefixes a sample. Both
  are legal queries with different rows.
* `drop_sort_below_limit_when_the_sort_is_by_a_constant` — **unsound in this engine**.
  With every key constant the sort is an arbitrary permutation, and arrow's
  `lexsort_to_indices` is *not* stable, so "the first `k`" after such a sort is not the
  positional first `k`. `prune_constant_sort_keys` refuses the same case for the same
  reason, and `sort_elimination_from_ordering` refuses a top-N over already-sorted input:
  "correctness over an extra rewrite".
"""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase, RuleCategory
from batcher.kyber.rules.fusion import _keys_match
from batcher.plan.expr_ir import Col
from batcher.plan.logical import (
    Distinct,
    Filter,
    Limit,
    LogicalPlan,
    Project,
    Sample,
    Sort,
    Union,
)
from batcher.plan.logical.aggregate import SortKeySpec

__all__ = [
    "collapse_topn_over_topn",
    "drop_limit_over_bounded_sample",
    "empty_limit_past_bounded_sample",
    "empty_topn_to_empty",
    "prune_input_of_empty_limit",
    "prune_sort_keys_after_unique_key",
    "push_topn_into_union",
    "topn_through_project",
]

# Single-input operators that neither change the schema nor produce rows of their own.
# Under a zero-limit the result is empty whatever they do, so they can be deleted. A
# `Project`/`Aggregate`/`Window`/`Union` is excluded: the first three change the schema
# (and the empty marker must keep it), and a `Union` may *promote* its branches' types,
# so no single branch's schema is the union's.
_EMPTY_TRANSPARENT = (Filter, Sort, Distinct, Sample)


@rule(name="topn_through_project", phase=Phase.FUSION, matches=(Sort,))
def topn_through_project(node: Sort, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Sort(Project(x), keys, limit=k)` → `Project(Sort(x, keys', limit=k), items)`.

    A projection is 1:1 and order-preserving, so the top-`k` of a projection is the
    projection of the top-`k` — and doing it in that order means the projection's
    expressions are evaluated on `k` rows instead of on the whole relation. This is the
    top-N analogue of `push_limit_through_project` (which cannot fire here: the `Sort`
    sits between the `Limit` and the `Project`, and by the time FUSION has folded the
    limit *into* the sort, PUSHDOWN is over).

    Fires only when every sort key is a bare column the projection merely passes through
    or renames (`alias = col(name)`), so the key can be rewritten to the underlying
    column with certainty. A key over a *computed* projection item is left alone: the
    expression would have to be duplicated below the sort, which is neither obviously
    cheaper nor obviously safe. Returns None otherwise, so the rule is idempotent.
    """
    if node.limit is None or not isinstance(node.input, Project):
        return None
    inner = node.input
    sources = {item.alias: item.expr for item in inner.items}
    keys: list[SortKeySpec] = []
    for key in node.keys:
        if not isinstance(key.expr, Col):
            return None
        source = sources.get(key.expr.name)
        if not isinstance(source, Col):
            return None
        keys.append(SortKeySpec(source, key.descending, key.nulls_first))
    return Project(Sort(inner.input, tuple(keys), node.limit), inner.items)


def _already_capped(branch: LogicalPlan, limit: int) -> bool:
    """Whether `branch` already ends in a top-`limit` (possibly under a projection)."""
    inner = branch.input if isinstance(branch, Project) else branch
    return isinstance(inner, Sort) and inner.limit == limit


@rule(name="push_topn_into_union", phase=Phase.FUSION, matches=(Sort,))
def push_topn_into_union(node: Sort, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Sort(UNION ALL(a, b, …), keys, limit=k)` → give each branch its own top-`k`.

    A row can only be in the global top-`k` if it is in its own branch's top-`k`: if `k`
    rows of its branch already rank ahead of it, so do those `k` rows globally. So
    capping each branch at `k` by the same ordering discards only rows the outer top-`k`
    could never return, and the outer `Sort(limit=k)` re-ranks the `k · branches`
    survivors — the classic distributed top-N (each partition ranks locally, the
    coordinator merges), here with the branches as the partitions. The win is that no
    branch ever materializes more than `k` rows for the union.

    Restricted to `UNION ALL` (a distinct union dedups *across* branches, so a row cut
    from one branch's top-`k` can still change the result). The `_already_capped` guard —
    which sees through the projection `topn_through_project` may have hoisted above a
    branch's sort — makes the rule fire once and then rest at a fixpoint.

    The argument above is exact for a *total* order. Where the keys tie across the `k`
    boundary, *which* of the tied rows a top-N returns is already engine-defined here —
    arrow's limited `lexsort_to_indices` is a partial (unstable) sort, and SQL does not
    specify it either — so the branch caps choose among the same tied rows the fused
    top-N was already choosing among, not a different set of key values. That is the
    envelope `topn_fusion` (`Limit(Sort(x))` → a partial sort) already occupies; this
    rule does not widen it, and it may not be widened further: nothing here may drop a
    row whose *key* could rank in the top-`k`.
    """
    inner = node.input
    if node.limit is None or node.limit <= 0 or not isinstance(inner, Union) or inner.distinct:
        return None
    if all(_already_capped(branch, node.limit) for branch in inner.inputs):
        return None
    branches = tuple(Sort(branch, node.keys, node.limit) for branch in inner.inputs)
    return Sort(Union(branches, distinct=False), node.keys, node.limit)


@rule(name="collapse_topn_over_topn", phase=Phase.FUSION, matches=(Sort,))
def collapse_topn_over_topn(node: Sort, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Sort(Sort(x, keys, limit=j), keys, limit=i)` → `Sort(x, keys, min(i, j))`.

    Two sorts by the *identical* ordering are one sort: re-sorting an already-sorted
    relation by the same keys changes nothing, so only the row caps matter, and the
    top-`i` of the top-`j` (by that same order) is the top-`min(i, j)`. A missing cap
    counts as unbounded, which covers the plain `Sort(Sort(x))` case as well as
    `Sort(Sort(x, limit=j))` (the outer sort of the inner's `j` rows is those same `j`
    rows in that same order).

    Requires structurally identical keys — same expressions, same direction, same null
    placement. Two *different* orderings do not collapse: the inner one breaks the outer
    one's ties, and dropping it would change which of the tied rows a top-N keeps.
    Returns None otherwise, so the rule is idempotent.
    """
    inner = node.input
    if not isinstance(inner, Sort) or not _keys_match(node.keys, inner.keys):
        return None
    limits = [limit for limit in (node.limit, inner.limit) if limit is not None]
    return Sort(inner.input, node.keys, min(limits) if limits else None)


@rule(
    name="empty_topn_to_empty",
    phase=Phase.FUSION,
    matches=(Sort,),
    category=RuleCategory.REWRITE,
)
def empty_topn_to_empty(node: Sort, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Sort(x, keys, limit=0)` → `Limit(x, 0)`, the canonical empty relation.

    `topn_fusion` folds `Limit(Sort(x), 0)` into a `Sort` with a zero cap — which is
    still empty, but no longer *looks* empty: `propagate_empty_relation` and the
    estimator's EXACT-zero recognize `Limit(x, 0)`, not `Sort(limit=0)`, so a query the
    optimizer had already proven empty went back to being an opaque sort. Restoring the
    marker lets the emptiness propagate up through everything above it again.

    Runs in SELECTION rather than FUSION on purpose: in FUSION it would trade turns with
    `topn_fusion` (which would re-fold the `Limit(x, 0)` straight back into the sort) and
    the phase's fixpoint would never converge.
    """
    if node.limit != 0:
        return None
    return Limit(node.input, 0)


@rule(name="prune_input_of_empty_limit", phase=Phase.REWRITE, matches=(Limit,))
def prune_input_of_empty_limit(node: Limit, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Limit(op(x), 0)` → `Limit(x, 0)` for a schema-preserving `op`.

    The canonical empty marker keeps its input only for its *schema*; nothing under it can
    contribute a row. So a `Filter`, `Sort`, `Distinct`, or `Sample` beneath a zero limit
    is pure dead work — sorting, deduplicating, or hashing rows that are all about to be
    discarded — and is deleted, while the schema (which each of them preserves exactly) is
    unchanged. The complement of `propagate_empty_relation`, which folds the marker
    *upward*: this one cleans out what is left below it.

    A schema-*changing* operator (`Project`, `Aggregate`, `Window`) is never pruned — the
    marker must keep reporting the right columns — and neither is a `Union`, whose output
    types are the promotion of its branches', so no single branch stands in for it.
    Recurses naturally through stacked operators (one layer per pass, then None).
    """
    if node.n != 0 or not isinstance(node.input, _EMPTY_TRANSPARENT):
        return None
    return Limit(node.input.input, 0)


@rule(name="drop_limit_over_bounded_sample", phase=Phase.REWRITE, matches=(Limit,))
def drop_limit_over_bounded_sample(node: Limit, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Limit(Sample(x, n=k), m, 0)` → `Sample(x, n=k)` when `m >= k`.

    A fixed-count sample emits at most `k` rows, so a zero-offset limit of `m >= k` keeps
    every one of them, in order — it is the identity. The bound is *structural* (it is on
    the plan node), which is what makes this rule necessary: the estimator can only say a
    sample yields `min(k, |x|)` rows with `DEFAULT` provenance, and every existing
    limit-dropping rule requires an EXACT count, so none of them can see it.

    Gated to `offset == 0` (an offset still skips rows) and `m >= 1` (`m == 0` is the
    empty marker, which must survive). Returns None otherwise, so the rule is idempotent.
    """
    inner = node.input
    if node.offset != 0 or node.n < 1 or not isinstance(inner, Sample) or inner.n is None:
        return None
    return inner if node.n >= inner.n else None


@rule(name="empty_limit_past_bounded_sample", phase=Phase.REWRITE, matches=(Limit,))
def empty_limit_past_bounded_sample(node: Limit, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Limit(Sample(x, n=k), m, offset)` → `Limit(…, 0)` when `offset >= k`.

    The offset sibling of `drop_limit_over_bounded_sample`, and sound for the same
    structural reason: a fixed-count sample emits **at most** `k` rows, so skipping `k` or
    more of them skips all of them — whatever the input's size, and however many rows the
    sample actually found. The result is provably empty, so it folds to the canonical
    marker `Limit(…, 0)`, which `prune_input_of_empty_limit` then strips and
    `propagate_empty_relation` hoists.

    Gated to a positive offset and an already-positive `n`, so it fires once and then
    leaves the `n == 0` marker alone (idempotent).
    """
    inner = node.input
    if node.offset < 1 or node.n < 1 or not isinstance(inner, Sample) or inner.n is None:
        return None
    return Limit(inner, 0) if node.offset >= inner.n else None


@rule(name="prune_sort_keys_after_unique_key", phase=Phase.REWRITE, matches=(Sort,))
def prune_sort_keys_after_unique_key(node: Sort, ctx: OptimizerContext) -> LogicalPlan | None:
    """Drop the `ORDER BY` keys that follow a provably-unique key — they never compare.

    A secondary sort key only ever breaks a tie on the keys before it. If some key holds a
    different value in every row, no two rows can tie on it, so every key *after* it is
    dead weight: the ordering (and the top-N it feeds) is identical without them, and each
    dropped key is one fewer column to row-encode and compare. `ORDER BY id, ts` over a
    primary key is exactly `ORDER BY id`.

    Uniqueness must be *proven*, never guessed: the key must be a bare column whose EXACT
    distinct count reaches the input's EXACT row count with a known-zero null count (two
    nulls would tie with each other, and a learned/sketch ndv is an estimate — neither may
    delete a real tiebreak). Direction and null placement are irrelevant to the pruned
    keys, since they are never consulted. Returns None when no key proves uniqueness or it
    is already the last one, so the rule is idempotent.
    """
    if ctx is None or len(node.keys) < 2:
        return None
    stats = ctx.estimator.estimate(node.input)
    if not stats.rows_exact:
        return None
    for index, key in enumerate(node.keys[:-1]):
        if not isinstance(key.expr, Col):
            continue
        stat = stats.column(key.expr.name)
        if (
            stat.ndv_is_exact
            and stat.ndv is not None
            and stat.null_count == 0
            and stat.ndv >= stats.rows
        ):
            return Sort(node.input, node.keys[: index + 1], node.limit)
    return None
