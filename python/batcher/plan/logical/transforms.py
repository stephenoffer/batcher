"""Plan transforms and predicates over `LogicalPlan` trees.

`remap_sources` shifts every `Scan.source_id` (used when appending a right side's
sources after the left's) and `share_sources` renumbers several branches onto one source
list, giving one relation one slot; `is_streamable` reports whether a plan is
partition-independent (only row-wise operators, no pipeline breaker);
`empty_result_schema` types a zero-batch result; `hoist_computed_keys` and
`project_columns` materialize a computed shuffle key as a hidden column and project it
away again, which is what gives an expression-keyed sort or window a distributed path.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import TypeVar

import pyarrow as pa

from batcher.plan.expr_ir import Col, Expr, Lit
from batcher.plan.logical.aggregate import Sort, SortKeySpec
from batcher.plan.logical.base import LogicalPlan
from batcher.plan.logical.join import Join
from batcher.plan.logical.relational import (
    Distinct,
    Filter,
    Limit,
    MapBatches,
    Project,
    Projection,
    Sample,
    Scan,
    Union,
)
from batcher.plan.logical.reshape import RowId, Unnest, Unpivot
from batcher.plan.schema import SchemaRef, placeholder_schema

#: A bound source object. `plan` is layer 1 and cannot name `io.source.Source` (layer 2),
#: and does not need to: `share_sources` only ever compares sources by identity.
_S = TypeVar("_S")

__all__ = [
    "constant_column_literal",
    "empty_result_schema",
    "hoist_computed_keys",
    "hoist_sort_key",
    "hoist_window_keys",
    "is_cartesian_key_pair",
    "is_partition_independent",
    "is_streamable",
    "passthrough_renames",
    "preserves_source_row_count",
    "project_columns",
    "rebuild_over_scan",
    "remap_sources",
    "share_sources",
    "split_streaming_tail",
    "streaming_fold_target",
]


def passthrough_renames(items: tuple) -> dict[str, str]:
    """A projection's `output alias -> source column`, for its bare-reference items only.

    The question "which of this projection's outputs *are* an input column, just possibly under
    another name?" — asked by every rule that needs to translate a predicate written against a
    projection's output into the names its input uses. Predicate pushdown asks it to move a
    conjunct below a `Project`, and the outer-join type rewrite asks it to see past the
    projection that a full outer join carries.

    Only an item whose expression is a bare `Col` is included, and callers depend on that
    exclusion rather than merely tolerating it: an item that *computes* something is not a
    reference to any single input column, so a fact about the output says nothing about an input.
    `coalesce(left_key, right_key)` is the case that makes it load-bearing — it is non-null
    wherever *either* side is, so "this output is not null" implies nothing about either input.

    Args:
        items: The projection's items (`Projection` records).

    Returns:
        The alias-to-source mapping, with computed items absent.
    """
    return {item.alias: item.expr.name for item in items if isinstance(item.expr, Col)}


# Sentinel distinguishing "the column is a known constant whose value is None" from
# "the column is not a provable constant". `constant_column_value` returns this when
# the column cannot be proven constant.
_NOT_CONSTANT = object()


def is_cartesian_key_pair(
    left: LogicalPlan, left_key: str, right: LogicalPlan, right_key: str
) -> bool:
    """Whether an equi-join key pair is a cartesian pseudo-edge (same constant on both sides).

    A key pair `left_key = right_key` where both columns are provably the *same* literal
    (the `__cross_key` a comma/cross join lowers to) is always true and connects nothing
    — it expresses a cartesian product, not a real join condition. Join reordering must
    not treat it as a graph edge (or it would happily build a cross product), and key
    derivation drops it once a real key is found. Anything not provably constant-on-both
    -sides returns False (treated as a genuine join edge).
    """
    lv = constant_column_value(left, left_key)
    if lv is _NOT_CONSTANT:
        return False
    rv = constant_column_value(right, right_key)
    if rv is _NOT_CONSTANT:
        return False
    return lv == rv


def constant_column_value(plan: LogicalPlan, column: str) -> object:
    """The literal value `column` provably holds in every output row, or `_NOT_CONSTANT`.

    Traces `column` down through value-preserving operators — a `Project` that binds it
    to a `Lit` (proof) or merely renames another column, and the row-preserving
    `Filter`/`Sort`/`Limit`/`Sample`/`Distinct` and inner `Join` — to the literal that
    defines it. Used to recognize synthetic constant join keys (e.g. the `__cross_key`
    a comma/cross join lowers to): a join key that is the same constant on both sides
    carries no information, so it is a cartesian pseudo-edge, not a real join condition.
    Anything it cannot prove returns `_NOT_CONSTANT` — never a guess.
    """
    if isinstance(plan, Project):
        for item in plan.items:
            if item.alias == column:
                if isinstance(item.expr, Lit):
                    return item.expr.value
                if isinstance(item.expr, Col):  # pure rename `column ← src`
                    return constant_column_value(plan.input, item.expr.name)
                return _NOT_CONSTANT
        return _NOT_CONSTANT
    if isinstance(plan, (Filter, Sort, Limit, Sample, Distinct)):
        return constant_column_value(plan.input, column)
    # Inner joins pass both sides' values through; semi/anti joins only *filter* the
    # left side's rows (their output is left-only), so a constant on the traced side
    # stays constant. Tracing through semi/anti lets a comma join's `__cross_key` be
    # recognized as a pseudo-edge even when a semi/anti join sits between it and the
    # join key referencing it (TPC-H Q18's `o_orderkey IN (…)`).
    if isinstance(plan, Join) and plan.join_type in ("inner", "semi", "anti"):
        for o in plan.output:
            if o.alias == column:
                child = plan.left if o.side == "left" else plan.right
                return constant_column_value(child, o.name)
        return _NOT_CONSTANT
    # A union's output rows are its branches' rows unchanged, so the column is constant
    # exactly when **every** branch proves the *same* constant. Branches are validated to
    # carry identical column names (`Union.__post_init__`), so each is traced by name.
    #
    # This arm is what makes the comma-join pseudo-key recognizable over a `UNION ALL`, and
    # its absence was expensive rather than merely incomplete. Projection pushdown moves the
    # synthetic `__cross_key = lit(1)` *into* each branch, so above the union the column is a
    # bare `Col` with no `Project` to prove it — `is_cartesian_key_pair` then read the pseudo
    # -edge as a genuine join key, `drop_redundant_cross_key` could not fire, and the join
    # ran on the composite `(__cross_key, real_key)`. That costs twice: it is a two-column
    # key, so the join takes `I64x2Keys` instead of `I64Keys` and forfeits the dense direct
    # map, and the constant column is materialized for every probe row. Measured on TPC-DS
    # sf1, the same 15-row-build join over `store_sales`: **5.7 ms** with a plain scan on the
    # probe side against **48.5 ms** with a two-branch `UNION ALL` — 8.5x, for a key that
    # matches every row.
    if isinstance(plan, Union):
        values = [constant_column_value(branch, column) for branch in plan.inputs]
        first = values[0]
        if first is _NOT_CONSTANT:
            return _NOT_CONSTANT
        for other in values[1:]:
            if other is _NOT_CONSTANT or other != first:
                return _NOT_CONSTANT
        return first
    return _NOT_CONSTANT


def constant_column_literal(plan: LogicalPlan, column: str) -> Lit | None:
    """The literal `column` provably holds in every output row, or None if not provable.

    The public form of the same proof [`is_cartesian_key_pair`] runs, for a caller that
    needs the *value* rather than the comparison: join reordering rebuilds a comma join's
    subtree from its leaves, and a synthetic constant column (the `__cross_key` a cross
    join lowers to) has no leaf to be traced back to — it is reproduced from its literal
    instead. Returns None for anything not provably constant, never a guess.

    Args:
        plan: The plan whose output `column` belongs to.
        column: The output column name.

    Returns:
        A `Lit` holding the proven value, or None.
    """
    value = constant_column_value(plan, column)
    return None if value is _NOT_CONSTANT else Lit(value)


def remap_sources(plan: LogicalPlan, offset: int) -> LogicalPlan:
    """Return a copy of `plan` with every `Scan.source_id` shifted by `offset`.

    Used when joining two datasets: the right side's sources are appended after
    the left's, so its scans must point past them.

    Args:
        plan: The plan to rewrite.
        offset: The amount to add to every `source_id`.

    Returns:
        A copy of `plan` whose scans point past `offset` earlier sources.
    """
    return _rewrite_source_ids(plan, lambda sid: sid + offset)


def share_sources(
    branches: Sequence[tuple[LogicalPlan, Sequence[_S]]],
) -> tuple[list[LogicalPlan], list[_S]]:
    """Put several branches on one source list, giving the *same* source one slot.

    `Dataset.union` concatenates its inputs' source lists, because in general two
    unioned datasets are unrelated and a source that appears in both is two relations
    that merely compare equal. This is the narrower operation for the callers that can
    prove otherwise -- the grouping levels of a `ROLLUP`/`CUBE`/`GROUPING SETS`, which
    are the *same* query over the *same* relations, differing only in what they group by.
    Concatenating there binds one relation once per level (a five-level rollup over three
    tables takes fifteen slots for three relations), and that alone is enough to stop
    plan-level common-subplan reuse recognizing the levels as sharing a subtree, so every
    level re-reads and re-joins the whole input.

    Sameness is **object identity**, never equality: two sources that compare equal may
    still be two independent handles, and merging those would change which relation a
    scan reads. Identity is the only test a caller can hand over without also handing
    over what its sources mean.

    Args:
        branches: Each branch's plan paired with the source list its scans index.

    Returns:
        The branches' plans rewritten onto one merged source list, and that list.
    """
    merged: list[_S] = []
    slot_of: dict[int, int] = {}
    plans: list[LogicalPlan] = []
    for plan, sources in branches:
        mapping: dict[int, int] = {}
        for index, source in enumerate(sources):
            slot = slot_of.get(id(source))
            if slot is None:
                slot = len(merged)
                slot_of[id(source)] = slot
                merged.append(source)
            mapping[index] = slot
        moved = any(index != slot for index, slot in mapping.items())
        plans.append(_rewrite_source_ids(plan, mapping.__getitem__) if moved else plan)
    return plans, merged


def _rewrite_source_ids(plan: LogicalPlan, renumber: Callable[[int], int]) -> LogicalPlan:
    """Return a copy of `plan` with every `Scan.source_id` passed through `renumber`.

    Only `Scan` carries a `source_id`; every other node is rebuilt generically with
    its remapped children by `transform_up`, so a new node type needs no edit here.
    The import is function-local because `plan.visitor` imports this module.
    """
    from batcher.plan.visitor import transform_up

    def move(node: LogicalPlan) -> LogicalPlan:
        if isinstance(node, Scan):
            # `replace`, not a fresh `Scan`: the source key is this scan's identity and
            # rebuilding without it would silently return the plan to the collided key.
            return dataclasses.replace(node, source_id=renumber(node.source_id))
        return node

    return transform_up(plan, move)


def empty_result_schema(plan: LogicalPlan, names: list[str]) -> pa.Schema:
    """The schema of a zero-batch result: the plan's inferred types, else null placeholders.

    A query that returns no rows still has a schema, and it must be the one a *matching*
    run would produce. Most shapes emit a zero-row batch (so the schema survives), but a
    few — notably `filter(<no match>).limit(k)`, where the limit stops before any batch is
    produced — emit none at all, and the null-typed fallback then handed the caller
    `i: null, v: null` for what a single matching row would have typed `int64`. That breaks
    `concat`, `write.parquet`, and any typed projection downstream.

    This lives in neutral `plan` rather than `api` because every executor must agree on it.
    `api` (relational), `dist` (spill), and `core` (streaming) each reached for an empty
    result and, being unable to import `api`, grew their own null-typed spelling — so the
    same empty query returned `null`-typed columns or `int64`-typed ones depending purely
    on which executor ran. That is the divergence this function exists to prevent, and it
    is why the helper is here and not one layer up.

    `available_schema()` infers the types for every relational shape; an opaque
    `map_batches` returns `None` and keeps the placeholders. The name guard keeps this
    strictly safer than a bare inference: a schema that disagrees with the caller's
    expected columns is discarded rather than trusted.

    Args:
        plan: The plan whose zero-batch result needs a schema.
        names: The column names the caller expects the result to carry.

    Returns:
        The plan's inferred Arrow schema when it matches `names`, else null placeholders.
    """
    schema = plan.available_schema()
    if schema is None or list(schema.arrow.names) != list(names):
        return placeholder_schema(names)
    return schema.arrow


def is_partition_independent(node: LogicalPlan) -> bool:
    """Whether this one node is a stateless, partition-independent transform.

    Running such a node on each partition (or each batch) and concatenating gives
    exactly the single-node result. `Unnest` (explode) and `Unpivot` (melt) multiply
    rows but hold no state, so they qualify.

    The subtle case, and the reason this predicate has one definition rather than one
    per execution path: a **fraction** `Sample` qualifies — a row is kept iff a seeded
    hash of its values falls under the fraction, a per-row predicate, so partitioning
    cannot change which rows survive. A **fixed-count** `Sample(n=)` does NOT: it keeps
    the `n` smallest-hash rows of the WHOLE relation, so running it per partition keeps
    `n` rows from *every* partition. Getting that line wrong does not raise — it returns
    a plausible wrong row count only under streaming or distribution.

    (The fraction hash reads every column, which is why `kyber.rules.projections` must
    not prune below a `Sample`; with pruning, a worker sampled a different column set
    than single-node did.)

    Args:
        node: The plan node to classify. Its input is not examined.

    Returns:
        True when the node itself is row-wise and partition-independent.
    """
    if isinstance(node, Sample):
        return node.n is None
    return isinstance(node, (Filter, Project, Unnest, Unpivot))


def preserves_source_row_count(node: LogicalPlan) -> bool:
    """Whether `node`'s output holds exactly one row per row of the source it scans.

    The question a *measurement* has to ask before it is written down as a fact about the
    source. `dist.executors.map` records the rows a distributed run produced under the
    source's identity, so the next run can size its partition count from a measured total
    instead of the blunt cluster-fill worker count. That is only a fact about the source when
    the plan that produced it neither drops rows nor adds them.

    It routinely does not. The same executor runs a per-partition `Distinct(limit=k)` (at most
    `workers x k` rows out of a billion-row table), a filtered scan, a fixed-count `Sample`,
    and a per-partition `Limit` — each of which recorded its own tiny output as the source's
    size. The next run then seeded from that, sized itself to one partition, and ran the whole
    query on one worker; the run after that recorded a smaller number still. The dispatcher
    already withholds the hub by hand at six call sites for exactly this reason, with a
    paragraph of comment at each; this is that rule stated once, where a new caller gets it
    without knowing the history.

    `Sort`, `Window` and `RowId` qualify (they reorder or widen, never resize). `Filter`,
    `Limit`, `Sample`, `Distinct`, `Aggregate`, `Unnest`, `Unpivot`, `Join`, `Union` and an
    opaque `MapBatches` do not.

    Args:
        node: The root of the per-partition plan whose output was measured.

    Returns:
        True when the measured row count is also the scanned source's row count.
    """
    from batcher.plan.logical.window import Window

    while True:
        if isinstance(node, Scan):
            return True
        if isinstance(node, (Project, Sort, Window, RowId)):
            node = node.input
            continue
        return False


def project_columns(plan: LogicalPlan, columns: Sequence[str]) -> Project:
    """A `Project` over `plan` selecting exactly `columns`, unchanged and in order.

    The projection every shuffle rewrite needs to restore a relation's user-visible
    schema after a hidden key column was materialized below it.

    Args:
        plan: The plan to project.
        columns: The column names to keep, in output order.

    Returns:
        A `Project` whose output columns are `columns`.
    """
    return Project(input=plan, items=tuple(Projection(alias=c, expr=Col(c)) for c in columns))


def hoist_sort_key(sort: Sort) -> tuple[Sort, tuple[str, ...]] | None:
    """Rewrite `ORDER BY <expr>, ...` so the LEADING key is a plain column.

    Returns `(sort', keep)` — the sort over a `Project` that materializes the computed
    leading key as a hidden column, plus the column names the result should carry (the
    original ones, so the hidden key is dropped) — or `None` when the leading key is already
    a column, which is the overwhelmingly common case and stays byte-identical.

    Every path that cuts a sort into ordered pieces range-partitions on the leading key's
    *values*, which it can only read from a column. Only the leading key is hoisted: the rest
    are evaluated by each piece's local sort, which needs no column.

    Args:
        sort: The sort to rewrite. Must carry at least one key.

    Returns:
        `(sort', keep)`, or `None` when no hoist is needed.
    """
    key = sort.keys[0]
    hoisted = hoist_computed_keys(sort.input, [key.expr], prefix="__sort_key")
    if hoisted is None:
        return None
    with_key, (hidden,) = hoisted
    keep = tuple(sort.input.available_columns())
    rewritten = dataclasses.replace(
        sort,
        input=with_key,
        keys=(
            SortKeySpec(hidden, descending=key.descending, nulls_first=key.nulls_first),
            *sort.keys[1:],
        ),
    )
    return rewritten, keep


def hoist_window_keys(window):
    """Rewrite `PARTITION BY <expr>, ...` so every partition key is a plain column.

    Returns `(window', keep)` — the window over a `Project` that materializes each computed
    partition key as a hidden column, plus the column names the result should carry — or
    `None` when every partition key is already a column.

    Every path that cuts a window into per-partition pieces reads the partition keys by
    column *position*, so a computed key such as `partition_by=[col("v") % 4]` has no such
    path unless it is materialized first. `keep` is the window's ORIGINAL output — its input
    columns plus the function aliases — so the hidden keys vanish and nothing else does.

    Args:
        window: The `Window` node to rewrite.

    Returns:
        `(window', keep)`, or `None` when no hoist is needed.
    """
    hoisted = hoist_computed_keys(window.input, window.partition_keys, prefix="__win_key")
    if hoisted is None:
        return None
    with_keys, keys = hoisted
    keep = tuple(window.available_columns())
    return dataclasses.replace(window, input=with_keys, partition_keys=keys), keep


def hoist_computed_keys(
    input_plan: LogicalPlan, keys: Sequence[Expr], *, prefix: str
) -> tuple[LogicalPlan, tuple[Expr, ...]] | None:
    """Materialize any computed shuffle key as a hidden column, so every key is a `Col`.

    A shuffle partitions on its keys' *values*, which it can only read from a column: the
    range partitioner splits on the leading sort key, and the hash partitioner reads the
    window's partition keys by column index. A key that is an expression — `sort(col("a") +
    col("b"))`, `partition_by=[col("v") % 4]` — therefore has no distributed path at all
    unless it is computed before the shuffle.

    Computing it once per row in the map prefix is exactly what the single-node operator
    does internally, so this is a rewrite and not an approximation: the hidden column is
    materialized below the breaker, drives the partitioning, and is projected away above it
    by `project_columns`. Callers own that second half, because only they know which
    columns their operator's output should carry.

    This lives in the neutral `plan` layer with one definition because the distributed and
    streaming paths both need it, and because the sort already had a private copy — a
    second one for the window is how the two spellings drift into disagreeing about
    shadowing or key order.

    Args:
        input_plan: The relation the keys are evaluated against.
        keys: The shuffle keys, in order.
        prefix: Base name for the hidden columns, e.g. ``"__sort_key"``. Underscores are
            appended until it shadows no existing column.

    Returns:
        `(input', keys')` where `input'` materializes the computed keys and every entry of
        `keys'` is a `Col`, or None when every key is already a column (the overwhelmingly
        common case, left byte-identical rather than wrapped in a no-op `Project`).
    """
    if all(isinstance(k, Col) for k in keys):
        return None

    columns = input_plan.available_columns()
    taken = set(columns)
    hidden: list[Projection] = []
    rewritten: list[Expr] = []
    for i, key in enumerate(keys):
        if isinstance(key, Col):
            rewritten.append(key)
            continue
        name = f"{prefix}_{i}"
        while name in taken:  # never shadow a user column
            name += "_"
        taken.add(name)
        hidden.append(Projection(alias=name, expr=key))
        rewritten.append(Col(name))

    with_keys = Project(
        input=input_plan,
        items=(*(Projection(alias=c, expr=Col(c)) for c in columns), *hidden),
    )
    return with_keys, tuple(rewritten)


def is_streamable(plan: LogicalPlan) -> bool:
    """Whether `plan` can be executed one source batch at a time in bounded memory.

    True iff every node is row-wise / partition-independent and there is no pipeline
    breaker (aggregate, sort, join, distinct, union, window, limit) that must see the
    whole input. Such plans are partition-independent, so running them per source batch
    yields exactly the same result as running them over the whole input.

    This is the recursive, whole-tree form of `is_partition_independent`, which owns the
    per-node rule (including the fraction-vs-fixed-count `Sample` distinction) so the
    streaming and distributed paths cannot drift apart on it.

    It admits one node the distributed classification does not: `MapBatches`. A batch UDF
    is row-wise *with respect to batching* — it is handed whole Arrow batches and cannot
    observe how the input was split — so streaming it per source batch is sound. The
    distributed path excludes it because it schedules UDFs through its own operator
    (GPU placement, actor pools, autobatching), not because the operator is stateful.

    Args:
        plan: The plan to classify.

    Returns:
        True when the whole plan can run one batch at a time.
    """
    if isinstance(plan, Scan):
        return True
    if isinstance(plan, MapBatches) or is_partition_independent(plan):
        return is_streamable(plan.input)
    return False


def streaming_fold_target(plan: LogicalPlan):
    """The mergeable `Aggregate` this streaming plan folds through, or None.

    One question — "is this operator `partial → combine → finalize` over a breaker-free
    input?" — asked by every path that runs a stateful stream: the single-node processor,
    the `iter_batches` router, and the distributed epoch gate. It lives in `plan` because
    those three are in three packages that may not import one another, and because the
    answer is a property of the plan rather than of who is running it.

    A whole-column `Distinct` is a group-by over every column and returns its `Aggregate`
    form. The single-node processor has always folded it that way (`Distinct.as_aggregate`),
    while the distributed gate tested `isinstance(plan, Aggregate)` and so refused
    `distinct()` on a cluster — a capability gap with no semantic cause, since the two paths
    would have been running the identical node. Answering both from here is what keeps them
    from disagreeing again.

    Three `Distinct` shapes are deliberately *not* folded:

    - a keyed `DISTINCT ON`, whose survivor carries columns the key does not determine, so
      no group-by expresses it (`as_aggregate` raises rather than approximating);
    - one carrying a fused `limit`, whose early exit an `Aggregate` cannot express;
    - one over a plan with a pipeline breaker beneath it, which no per-batch fold reaches.

    Answers for a **bare** fold only. A plan carrying row-wise work above the aggregate is a
    mergeable fold too, but running it needs that tail applied to each snapshot, so it is
    `split_streaming_tail` that reports it — and this returns None rather than hand a caller
    that cannot apply a tail a plan that has one.

    Args:
        plan: The top-level streaming plan.

    Returns:
        The `Aggregate` to fold, None when this plan is not a mergeable fold, and None when
        it is one carrying a tail (see `split_streaming_tail`).
    """
    split = split_streaming_tail(plan)
    # Only a *bare* fold, so this keeps meaning exactly what it did before the tail split
    # existed: a caller that cannot apply a tail must not be handed a plan that has one.
    return split[1] if split is not None and not split[0] else None


def rebuild_over_scan(nodes: Sequence[LogicalPlan], schema: pa.Schema) -> LogicalPlan:
    """Re-root a chain of single-input nodes on a fresh `Scan` of `schema`.

    The half that pairs with a split: `split_streaming_tail` (and the distributed
    dispatcher's `_split_at`) hand back the operators *above* a breaker, and both paths then
    need to run exactly those operators over the breaker's assembled result. That result is
    a table, so the chain is re-rooted on a scan of it.

    Shared rather than written twice because the two callers are in packages that may not
    import each other (`core` folds a stream, `dist` re-applies above a distributed
    breaker), and a second copy is precisely how the streaming and distributed tails would
    come to disagree about what "the operators above" means.

    Args:
        nodes: The chain, outermost first — the order a split returns it in.
        schema: The assembled result's schema, which the new scan reads.

    Returns:
        The outermost node of the rebuilt chain, or the bare `Scan` when `nodes` is empty.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> import pyarrow as pa
            >>> from batcher.plan.logical import rebuild_over_scan, split_streaming_tail
            >>> ds = bt.from_pydict({"k": ["a"], "v": [1]})
            >>> tail, _ = split_streaming_tail(
            ...     ds.group_by("k").agg(s=bt.col("v").sum()).select("s")._plan
            ... )
            >>> schema = pa.schema([("k", pa.string()), ("s", pa.int64())])
            >>> rebuilt = rebuild_over_scan(tail, schema)
            >>> type(rebuilt).__name__
            'Project'
    """
    plan: LogicalPlan = Scan(0, SchemaRef.from_arrow(schema))
    for node in reversed(list(nodes)):  # innermost (closest to the breaker) first
        plan = dataclasses.replace(node, input=plan)
    return plan


def split_streaming_tail(plan: LogicalPlan):
    """The row-wise tail above a streaming plan's mergeable fold, plus that fold.

    The streaming counterpart of what the distributed dispatcher already does with
    `_split_at` and `_apply_above`: a pipeline breaker is distributed (or folded) on its
    own, and the row-wise operators *above* it are re-applied to its assembled result.
    Applying a row-wise node to the fold's full running snapshot is exactly what the batch
    plan computes over the whole input, because such a node's output for a row depends on
    that row alone.

    Without this, every shape with post-aggregate work was refused from a streaming sink —
    ``group_by(k).agg(...).select(...)``, a HAVING filter, and *every* expression over
    aggregates (``sum(x) / count()``, ``max(v) - min(v)``, `regr_slope`), because those
    lower to a `Project` over the `Aggregate` rather than to one node. Batch ran all of
    them; streaming answered "this plan cannot be streamed to a sink", which reads as a
    missing operator rather than the missing projection it was.

    The tail is exactly `is_partition_independent`, the predicate the distributed path
    already shares with this one — so a node becomes streamable above a fold at the same
    moment it becomes safe to run per partition, and there is no second list to drift.
    `Limit` and `Sort` are deliberately absent from it: neither is row-wise, and on a
    *running* result neither has a batch meaning to match.

    Args:
        plan: The top-level streaming plan.

    Returns:
        ``(tail, aggregate)`` with `tail` outermost-first, or None when this plan is not
        a mergeable fold. An empty `tail` is the bare-aggregate case.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.plan.logical import split_streaming_tail
            >>> ds = bt.from_pydict({"k": ["a"], "v": [1]})
            >>> tail, agg = split_streaming_tail(
            ...     ds.group_by("k").agg(s=bt.col("v").sum()).select("s")._plan
            ... )
            >>> [type(n).__name__ for n in tail], type(agg).__name__
            (['Project'], 'Aggregate')
    """
    from batcher.plan.logical.aggregate import Aggregate

    tail: list[LogicalPlan] = []
    node = plan
    while is_partition_independent(node):
        tail.append(node)
        node = node.input
    if isinstance(node, Aggregate):
        return (tuple(tail), node) if is_streamable(node.input) else None
    if isinstance(node, Distinct):
        if node.keys or node.limit is not None or not is_streamable(node.input):
            return None
        return tuple(tail), node.as_aggregate()
    return None
