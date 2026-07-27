"""Projection, ordering, and scan/schema simplifications — local, always-correct.

Each rule here is a node-local `@rule`: it returns a rewritten node or `None` to
leave it unchanged; the driver supplies bottom-up traversal and fixpoint iteration.
They are the small, certainly-correct cleanups that complement the whole-plan column
pruner (`projection_rewrite`) and the metadata-driven zone-map rules — never
duplicating them, only handling the *local* shapes they leave behind:

- redundant ordering: a sort feeding an order-indifferent `Sample` is dead, and
  duplicate or identity-cast sort keys add nothing;
- schema-provable null checks over a NOT-NULL source column (drop always-true ones;
  fold an impossible one to an empty relation) — using only declared nullability,
  never statistics;
- degenerate/composable `Sample` bounds (`n = 0` is empty; a full `fraction` is the
  identity; same-seed nested samples compose), anchored to the engine's exact sampling
  semantics;
- projection/expression cleanups the merge/identity/fold rules decline: inlining a
  multiply-referenced *rename* (free to duplicate) and dropping a cast of a column to
  the type it already has (in projections and filters alike).

All are unconditionally semantics-preserving (result multiset unchanged); only the
plan shape changes.
"""

from __future__ import annotations

import dataclasses

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.plan.expr_ir import Cast, Col, Expr, IsNotNull, IsNull, Not
from batcher.plan.expr_ir.walk import column_occurrence_counts
from batcher.plan.expr_rewrite import (
    combine_conjuncts,
    expr_key,
    split_conjuncts,
    substitute_columns,
    transform_expr_up,
)
from batcher.plan.logical import (
    Distinct,
    Filter,
    Limit,
    LogicalPlan,
    Project,
    Projection,
    Sample,
    Scan,
    Sort,
    SortKeySpec,
)
from batcher.plan.schema import SchemaRef
from batcher.plan.types import DTYPE_REGISTRY, infer_type

__all__ = [
    "dedupe_sort_keys",
    "drop_always_true_null_check",
    "drop_self_cast_in_filter",
    "drop_self_cast_in_projection",
    "drop_self_cast_in_sort_key",
    "eliminate_sort_before_sample",
    "empty_on_impossible_null_check",
    "empty_sample_n",
    "fold_nested_sample_same_seed",
    "identity_full_sample",
    "merge_projection_renames",
]


# --- ordering: drop work the ordering makes irrelevant -----------------------


@rule(name="eliminate_sort_before_sample", phase=Phase.NORMALIZE, matches=(Sample,))
def eliminate_sort_before_sample(node: Sample, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Sample(Sort(x))` → `Sample(x)`. A row is kept by a stable hash of its *values*
    (fraction mode) or by being among the `n` smallest such hashes — neither depends on
    input order, so the sampled multiset is identical and the sort is pure overhead.
    Skipped when the sort carries a `limit` (a top-N changes the input set that is
    sampled)."""
    inner = node.input
    if isinstance(inner, Sort) and inner.limit is None:
        return dataclasses.replace(node, input=inner.input)
    return None


def _sort_key_sig(key: object) -> tuple:
    """A structural signature for a sort key (lowered IR + direction + null placement).

    Uses `expr_key`'s canonical *string* rather than the raw IR dict, which makes the
    signature hashable — so `dedupe_sort_keys` can hold the seen set in a `set` instead of
    scanning a list and comparing whole dicts, one full comparison per key already kept.
    """
    return (expr_key(key.expr), key.descending, key.nulls_first)


@rule(name="dedupe_sort_keys", phase=Phase.NORMALIZE, matches=(Sort,))
def dedupe_sort_keys(node: Sort, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Drop a sort key identical to an earlier one (`ORDER BY a, b, a` → `ORDER BY a,
    b`). Once rows are ordered by a key, a later key equal to it can only compare within
    ties on that key — where it is, by construction, also equal — so it never
    discriminates. Removing exact-duplicate keys leaves the ordering (and the result)
    unchanged. Returns None when every key is unique, keeping the rule idempotent."""
    seen: set[tuple] = set()
    kept = []
    for key in node.keys:
        sig = _sort_key_sig(key)
        if sig in seen:
            continue
        seen.add(sig)
        kept.append(key)
    if len(kept) == len(node.keys):
        return None
    return Sort(node.input, tuple(kept), node.limit)


# --- scan/schema: null checks over a declared NOT-NULL column ----------------


def _non_nullable_cols(node: LogicalPlan) -> frozenset[str]:
    """Columns provably non-nullable at `node`'s output, from the source schema alone.

    Traces down through row/column-preserving unary operators (`Filter`/`Sort`/`Limit`/
    `Sample`/`Distinct` — none introduces a null into an existing column, nor renames
    one) to the originating `Scan`, and reads the declared nullability of its *raw*
    source schema (`Scan.available_schema` widens types and drops nullability, so the
    raw `Scan.schema` is used). A `Project`/`Join`/`Aggregate`/… on the way down stops
    the trace conservatively — the column may be computed or null-extended there."""
    while isinstance(node, (Filter, Sort, Limit, Sample, Distinct)):
        node = node.input
    if isinstance(node, Scan):
        return frozenset(f.name for f in node.schema.arrow if not f.nullable)
    return frozenset()


def _null_check_col(expr: Expr, *, want_null: bool) -> str | None:
    """The column name of a bare `IS NULL` (`want_null`) / `IS NOT NULL` check, else None.

    Handles both spellings of each: `IsNull(col)` / `Not(IsNotNull(col))` for the null
    check, and `IsNotNull(col)` / `Not(IsNull(col))` for the not-null check."""
    positive = IsNull if want_null else IsNotNull
    negative = IsNotNull if want_null else IsNull
    if isinstance(expr, positive) and isinstance(expr.input, Col):
        return expr.input.name
    if isinstance(expr, Not) and isinstance(expr.input, negative):
        inner = expr.input.input
        if isinstance(inner, Col):
            return inner.name
    return None


@rule(name="drop_always_true_null_check", phase=Phase.NORMALIZE, matches=(Filter,))
def drop_always_true_null_check(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Drop `col IS NOT NULL` conjuncts over a column the schema declares NOT NULL.

    Such a check yields a plain `true` on every row (the column has no nulls), so it is
    an identity conjunct: removing it from the top-level `AND` leaves the result
    unchanged. If it was the *only* conjunct, the whole filter is dead and the input
    flows through. Uses only declared nullability (never statistics), so it is exact —
    a complement to `predicate_infer`'s sibling-conjunct inference. Returns None when
    nothing is removable, keeping the rule idempotent."""
    non_null = _non_nullable_cols(node.input)
    if not non_null:
        return None
    conjuncts = split_conjuncts(node.predicate)
    kept = [c for c in conjuncts if _null_check_col(c, want_null=False) not in non_null]
    if len(kept) == len(conjuncts):
        return None
    if not kept:
        return node.input
    return Filter(node.input, combine_conjuncts(kept))


@rule(name="empty_on_impossible_null_check", phase=Phase.SELECTION, matches=(Filter,))
def empty_on_impossible_null_check(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Fold a filter to an empty relation when a conjunct is `col IS NULL` over a column
    the schema declares NOT NULL.

    That check is `false` on every row (no nulls exist), so the conjunction — and the
    whole filter — matches nothing. It becomes the canonical empty marker `Limit(x, 0)`,
    which then lets row counts propagate and the empty-relation rules fold operators
    above. Schema-exact (declared nullability, no statistics); returns None otherwise."""
    non_null = _non_nullable_cols(node.input)
    if not non_null:
        return None
    for conj in split_conjuncts(node.predicate):
        if _null_check_col(conj, want_null=True) in non_null:
            return Limit(node.input, 0)
    return None


# --- sample: degenerate/composable bounds, per the engine's semantics --------


@rule(name="empty_sample_n", phase=Phase.SELECTION, matches=(Sample,))
def empty_sample_n(node: Sample, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Sample(x, n=0)` → empty. A fixed-count sample of zero rows keeps nothing (the
    engine returns no rows for `n == 0`), so it is the empty relation `Limit(x, 0)` —
    schema-preserving, and it lets the emptiness propagate upward. Only the exact `n = 0`
    case is folded (a fractional sample can keep a row whose hash is zero, so
    `fraction = 0.0` is *not* provably empty)."""
    if node.n is not None and node.n == 0:
        return Limit(node.input, 0)
    return None


@rule(name="identity_full_sample", phase=Phase.REWRITE, matches=(Sample,))
def identity_full_sample(node: Sample, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Sample(x, fraction=1.0)` → `x`. A full-fraction sample keeps every row (the
    engine short-circuits at `fraction >= 1.0`), so the operator is a no-op. Restricted
    to fraction mode (`n is None`); the fixed-count path is handled by `empty_sample_n`
    and the `total <= n` keep-everything case is a runtime decision, not a plan shape."""
    if node.n is None and node.fraction >= 1.0:
        return node.input
    return None


@rule(name="fold_nested_sample_same_seed", phase=Phase.REWRITE, matches=(Sample,))
def fold_nested_sample_same_seed(node: Sample, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Sample(Sample(x, f1, s), f2, s)` → `Sample(x, min(f1, f2), s)` when both are
    fraction-mode samples with the *same* seed.

    A fraction sample keeps a row iff a seeded hash of its values is under the fraction's
    threshold. With an identical seed the two samples compute the same per-row hash, so a
    row survives both iff its hash is under *both* thresholds — i.e. under `min(f1, f2)`
    (thresholds are monotonic in the fraction). Two passes collapse to one with an
    identical result. Conservative: distinct or auto-generated seeds (which differ) and
    the fixed-count mode are left untouched."""
    inner = node.input
    if isinstance(inner, Sample) and node.n is None and inner.n is None and node.seed == inner.seed:
        return Sample(inner.input, min(node.fraction, inner.fraction), node.seed)
    return None


# --- projection/expression cleanups the merge/fold rules decline -------------


@rule(name="merge_projection_renames", phase=Phase.NORMALIZE, matches=(Project,))
def merge_projection_renames(node: Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Project(Project(x))` → one `Project(x)` even when a *renamed* inner column is
    referenced by the outer projection more than once.

    `merge_projections` conservatively declines any inner column referenced twice or more
    (inlining a *computed* one would duplicate its work). But a pass-through/rename inner
    item is a bare `Col`, and inlining a bare column any number of times adds zero
    compute — it is just a renaming. This rule handles exactly that case the general
    merge leaves behind: it fires only when some inner column is multiply-referenced
    (so the general merge declined) and *every* such multiply-referenced inner item is a
    bare column (so the inline is free), then folds the two projections into one."""
    inner = node.input
    if not isinstance(inner, Project):
        return None
    counts = column_occurrence_counts([it.expr for it in node.items])
    multi = [it for it in inner.items if counts.get(it.alias, 0) > 1]
    if not multi:
        return None  # the ≤1-reference case is `merge_projections`' job
    if any(not isinstance(it.expr, Col) for it in multi):
        return None  # a computed inner column referenced >1 — inlining would duplicate work
    inner_map = {it.alias: it.expr for it in inner.items}
    new_items = tuple(
        Projection(it.alias, substitute_columns(it.expr, inner_map)) for it in node.items
    )
    return Project(inner.input, new_items)


def _strip_self_cast(expr: Expr, schema: SchemaRef) -> Expr:
    """`cast(col(c), T)` → `col(c)` when `c` already has type `T`, else `expr` unchanged."""
    if isinstance(expr, Cast) and isinstance(expr.input, Col):
        target = DTYPE_REGISTRY.get(expr.dtype)
        if target is not None and infer_type(expr.input, schema) == target:
            return expr.input
    return expr


@rule(name="drop_self_cast_in_projection", phase=Phase.NORMALIZE, matches=(Project,))
def drop_self_cast_in_projection(node: Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Drop a cast of a column to the type it already has, inside a projection.

    `cast(col(c), T)` where `c` is already `T` (per the input schema — which reflects the
    engine's post-FFI widened types) is the identity, whether a strict or a try-cast.
    Removing it leaves the value bit-identical and reduces the item to a plain column
    reference. Rewrites every projection expression bottom-up; returns None when no item
    changes, keeping the rule idempotent."""
    schema = node.input.available_schema()
    if schema is None:
        return None
    new_items = []
    changed = False
    for item in node.items:
        stripped = transform_expr_up(item.expr, lambda e: _strip_self_cast(e, schema))
        if stripped.to_ir() != item.expr.to_ir():
            changed = True
        new_items.append(Projection(item.alias, stripped))
    if not changed:
        return None
    return dataclasses.replace(node, items=tuple(new_items))


@rule(name="drop_self_cast_in_filter", phase=Phase.NORMALIZE, matches=(Filter,))
def drop_self_cast_in_filter(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Drop a cast of a column to the type it already has, inside a filter predicate.

    The same identity-cast simplification as `drop_self_cast_in_projection`, applied to
    the predicate: `cast(col(c), T)` where `c` is already `T` is a no-op, so stripping it
    exposes a plain `col(c)` that zone-map pruning and predicate pushdown (which look for
    bare `col OP literal` shapes) can then use. Returns None when the predicate is
    unchanged, keeping the rule idempotent."""
    schema = node.input.available_schema()
    if schema is None:
        return None
    new_pred = transform_expr_up(node.predicate, lambda e: _strip_self_cast(e, schema))
    if new_pred.to_ir() == node.predicate.to_ir():
        return None
    return Filter(node.input, new_pred)


@rule(name="drop_self_cast_in_sort_key", phase=Phase.NORMALIZE, matches=(Sort,))
def drop_self_cast_in_sort_key(node: Sort, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Drop a cast of a column to the type it already has, inside a sort key.

    `cast(col(c), T)` where `c` is already `T` is the identity, so ordering by it is
    identical to ordering by `col(c)` directly (same values, same comparisons) — the
    order is unchanged even under a stable sort. Stripping it also lets ordering analysis
    recognise the plain column. Preserves each key's direction/null placement and the
    sort node itself (only the key expressions change); returns None when nothing
    changes, keeping the rule idempotent."""
    schema = node.input.available_schema()
    if schema is None:
        return None
    new_keys = []
    changed = False
    for key in node.keys:
        stripped = transform_expr_up(key.expr, lambda e: _strip_self_cast(e, schema))
        if stripped.to_ir() != key.expr.to_ir():
            changed = True
            new_keys.append(SortKeySpec(stripped, key.descending, key.nulls_first))
        else:
            new_keys.append(key)
    if not changed:
        return None
    return Sort(node.input, tuple(new_keys), node.limit)
