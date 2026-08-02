"""Plan transforms and predicates over `LogicalPlan` trees.

`remap_sources` shifts every `Scan.source_id` (used when appending a right side's
sources after the left's); `is_streamable` reports whether a plan is
partition-independent (only row-wise operators, no pipeline breaker);
`empty_result_schema` types a zero-batch result; `hoist_computed_keys` and
`project_columns` materialize a computed shuffle key as a hidden column and project it
away again, which is what gives an expression-keyed sort or window a distributed path.
"""

from __future__ import annotations

from collections.abc import Sequence

import pyarrow as pa

from batcher.plan.expr_ir import Col, Expr, Lit
from batcher.plan.logical.aggregate import Sort
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
)
from batcher.plan.logical.reshape import Unnest, Unpivot
from batcher.plan.schema import placeholder_schema

__all__ = [
    "empty_result_schema",
    "hoist_computed_keys",
    "is_cartesian_key_pair",
    "is_partition_independent",
    "is_streamable",
    "passthrough_renames",
    "project_columns",
    "remap_sources",
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
    return _NOT_CONSTANT


def remap_sources(plan: LogicalPlan, offset: int) -> LogicalPlan:
    """Return a copy of `plan` with every `Scan.source_id` shifted by `offset`.

    Used when joining two datasets: the right side's sources are appended after
    the left's, so its scans must point past them.

    Only `Scan` carries a `source_id`; every other node is rebuilt generically with
    its remapped children by `transform_up`, so a new node type needs no edit here.
    The import is function-local because `plan.visitor` imports this module.
    """
    from batcher.plan.visitor import transform_up

    def shift(node: LogicalPlan) -> LogicalPlan:
        if isinstance(node, Scan):
            return Scan(node.source_id + offset, node.schema)
        return node

    return transform_up(plan, shift)


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
