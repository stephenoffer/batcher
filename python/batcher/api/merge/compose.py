"""Compose a full SQL ``MERGE`` out of relational algebra — no new IR, so it distributes.

A ``MERGE`` looks like a new operator, and building it as one would be a mistake: it would
need its own parallel path, its own distributed path, its own spill story, and its own
correctness proof. It needs none of those, because it *is* already expressible in the
algebra the engine has. Every row of the result comes from exactly one of three disjoint
populations, each a join away:

    matched               = target ⋈  source     (on the merge keys)
    not matched           = source  ▷ target     (anti — a new key)
    not matched by source = target  ▷ source     (anti — a key the source dropped)

and the ordered ``WHEN`` clauses within a population are exactly a chained ``CASE`` (first
true branch wins — `Expr`'s `Case` and SQL's ``MERGE`` agree on this, including that a NULL
condition is *not* true and falls through). The result is the union of the three.

So the whole of ``MERGE`` is joins + ``CASE`` + ``union``, which means it inherits — for
free, and with no second semantics to keep in sync — the morsel-parallel path, the shuffle
join, the spill, and the distributed executor. That is invariant 7 (`single-node ==
distributed via mergeable algebra`) doing real work: the reason a merge runs on a cluster
at all is that nothing here knows whether it is on one.

## Deletes are a column, not a control-flow branch

A clause chain produces two things per row: the new value of each target column, and
whether the row survives. Both are `CASE`s over the *same* ordered conditions, so they
cannot disagree about which clause won. The survival flag becomes a reserved boolean
column, the row is filtered on it, and the flag is projected away.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.api.merge.clauses import (
    MATCHED,
    NOT_MATCHED,
    NOT_MATCHED_BY_SOURCE,
    SOURCE_PREFIX,
    MergeClause,
    source_name,
    validate_clause,
)
from batcher.plan.expr_ir import Col, Expr, lit, nullif, when

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pyarrow as pa

    from batcher.api.dataset import Dataset

__all__ = ["compose_merge", "merge_output_columns"]

# The survival flag a clause chain writes and the final projection drops.
_KEEP = "__bc_merge_keep"

# Arrow type → the `cast` name that produces a NULL of that type, for a target column an
# INSERT clause does not write (SQL leaves it NULL). `CAST_DTYPES` is deliberately small;
# a type outside it gets an explicit error rather than a silently wrong type.
_NULL_CASTS: list[tuple[str, str]] = [
    ("is_boolean", "bool"),
    ("is_string", "string"),
    ("is_large_string", "string"),
    ("is_date", "date32"),
    ("is_timestamp", "timestamp"),
    ("is_floating", "float64"),
    ("is_integer", "int64"),
]


def merge_output_columns(target: Dataset) -> list[str]:
    """The merge's output columns — always the target's, in the target's order.

    A ``MERGE`` writes the target table; it never widens or reorders it. Pinning the output
    to the target's schema is what lets the three populations `union` at all.
    """
    return list(target.columns)


def compose_merge(
    source: Dataset,
    target: Dataset,
    keys: Sequence[str],
    clauses: Sequence[MergeClause],
) -> Dataset:
    """Compose the target's post-merge state as a single lazy relation.

    The returned `Dataset` is an ordinary plan — it optimizes, parallelizes, spills, and
    distributes like any other, and it is what both the single-node and the distributed
    merge execute.

    Args:
        source: The change set being merged in.
        target: The table being merged into (supplies the output schema).
        keys: The columns matching a source row to a target row.
        clauses: The ordered ``WHEN …`` clauses.

    Returns:
        The merged relation, with exactly the target's columns.
    """
    columns = merge_output_columns(target)
    keys = list(keys)
    _validate(source, keys, clauses, columns)

    matched = [c for c in clauses if c.kind == MATCHED]
    inserts = [c for c in clauses if c.kind == NOT_MATCHED]
    by_source = [c for c in clauses if c.kind == NOT_MATCHED_BY_SOURCE]

    # Every source column moves to a reserved name, so a source and a target column of the
    # same name coexist in the joined relation and neither is shadowed or suffixed.
    prefixed = source.select(**{source_name(c): Col(c) for c in source.columns})
    source_keys = [source_name(k) for k in keys]

    parts: list[Dataset] = []
    parts.extend(_matched_part(prefixed, target, keys, source_keys, matched, columns))
    parts.extend(_by_source_part(prefixed, target, keys, source_keys, by_source, columns))
    parts.extend(_insert_part(prefixed, target, keys, source_keys, inserts, columns, source))

    if not parts:
        # Every population is deleted or dropped: the target ends up empty. `limit(0)` keeps
        # the schema, which an empty `union` could not.
        return target.select(*columns).limit(0)
    result = parts[0]
    for part in parts[1:]:
        result = result.union(part)
    return result


def _validate(
    source: Dataset,
    keys: list[str],
    clauses: Sequence[MergeClause],
    columns: list[str],
) -> None:
    if not keys:
        raise PlanError("merge(): `on` requires at least one key column")
    if not clauses:
        raise PlanError(
            "merge(): no WHEN clauses — a merge with no clauses would rewrite the target "
            "unchanged. Add when_matched()/when_not_matched()/when_not_matched_by_source()."
        )
    # Each side's names go into a set once: a linear scan per key made validating a
    # multi-key merge against a wide table cost keys x width comparisons, and
    # `source.columns` rebuilds its list on every access.
    target_names = set(columns)
    source_names = set(source.columns)
    missing_target = [k for k in keys if k not in target_names]
    if missing_target:
        raise PlanError(f"merge(): key column(s) {missing_target} are not in the target")
    missing_source = [k for k in keys if k not in source_names]
    if missing_source:
        raise PlanError(f"merge(): key column(s) {missing_source} are not in the source")

    reserved = sorted(c for c in (*source.columns, *columns) if c.startswith(SOURCE_PREFIX))
    if reserved:
        raise PlanError(
            f"merge(): {reserved} use the reserved {SOURCE_PREFIX!r} prefix; rename them"
        )
    if _KEEP in source.columns or _KEEP in columns:
        raise PlanError(f"merge(): {_KEEP!r} is a reserved column name; rename it")

    for clause in clauses:
        validate_clause(clause, columns)
        if clause.values is None and not clause.is_delete:
            # `UPDATE SET *` / `INSERT *` reads every target column from the source.
            absent = [c for c in columns if c not in source.columns]
            if absent:
                raise PlanError(
                    f"merge(): {clause.kind} clause writes all columns from the source, but "
                    f"the source has no column(s) {absent}. List the columns explicitly."
                )


def _matched_part(
    prefixed: Dataset,
    target: Dataset,
    keys: list[str],
    source_keys: list[str],
    clauses: Sequence[MergeClause],
    columns: list[str],
) -> list[Dataset]:
    """Target rows whose key is in the source, after their clause chain."""
    if not clauses:
        # No matched clause ⇒ a matched row is untouched. A *semi* join says exactly that
        # and cannot fan out, so it is both the right semantics and the cheaper plan.
        untouched = target.join(
            prefixed.select(*source_keys).distinct(),
            left_on=keys,
            right_on=source_keys,
            how="semi",
        )
        return [untouched.select(*columns)]

    joined = target.join(prefixed, left_on=keys, right_on=source_keys, how="inner")
    # The join emits each key once, under the target's name; re-materialize the source's
    # spelling so `source_col(key)` resolves in a clause. On an equijoin they are equal.
    joined = joined.with_columns(**{source_name(k): Col(k) for k in keys})
    return _apply(joined, clauses, columns, default=Col, default_keep=True)


def _by_source_part(
    prefixed: Dataset,
    target: Dataset,
    keys: list[str],
    source_keys: list[str],
    clauses: Sequence[MergeClause],
    columns: list[str],
) -> list[Dataset]:
    """Target rows whose key is absent from the source, after their clause chain."""
    unmatched = target.join(
        prefixed.select(*source_keys).distinct(), left_on=keys, right_on=source_keys, how="anti"
    )
    if not clauses:
        return [unmatched.select(*columns)]  # untouched — the survivors of a plain upsert
    return _apply(unmatched, clauses, columns, default=Col, default_keep=True)


def _insert_part(
    prefixed: Dataset,
    target: Dataset,
    keys: list[str],
    source_keys: list[str],
    clauses: Sequence[MergeClause],
    columns: list[str],
    source: Dataset,
) -> list[Dataset]:
    """Source rows whose key is absent from the target, after their clause chain."""
    if not clauses:
        return []  # WHEN NOT MATCHED omitted ⇒ new keys are ignored, not inserted
    new_rows = prefixed.join(
        target.select(*keys).distinct(), left_on=source_keys, right_on=keys, how="anti"
    )
    schema = target.schema
    default = _insert_default(schema, source)
    # A row matching no insert clause is NOT inserted — hence `default_keep=False`, unlike
    # the two target-side populations where an unmatched row survives untouched.
    return _apply(new_rows, clauses, columns, default=default, default_keep=False)


def _insert_default(schema: pa.Schema, source: Dataset) -> Any:
    """The value an INSERT clause gives a target column it does not write: a typed NULL."""

    def default(column: str) -> Expr:
        if column in source.columns:
            # A NULL of the source column's own type — no cast vocabulary needed.
            return nullif(Col(source_name(column)), Col(source_name(column)))
        return _null_of(schema, column)

    return default


def _null_of(schema: pa.Schema, column: str) -> Expr:
    """A NULL literal typed as the target's `column`."""
    import pyarrow as pa

    dtype = schema.field(column).type
    for predicate, cast_name in _NULL_CASTS:
        if getattr(pa.types, predicate)(dtype):
            return nullif(lit(0), lit(0)).cast(cast_name)
    raise PlanError(
        f"merge(): the insert clause does not write target column {column!r} (type {dtype}), "
        "and a NULL of that type cannot be constructed. Write the column explicitly."
    )


def _apply(
    relation: Dataset,
    clauses: Sequence[MergeClause],
    columns: list[str],
    default: Any,
    default_keep: bool,
) -> list[Dataset]:
    """Run an ordered clause chain over `relation`, yielding rows with exactly `columns`.

    Each target column and the survival flag become chained `CASE`s over the *same* ordered
    conditions, so every one of them agrees on which clause won. `default` supplies the
    value a column keeps when no clause fires; `default_keep` says whether such a row
    survives at all (true for the target-side populations, which are left untouched; false
    for inserts, where matching no clause means the row is simply not inserted).
    """
    projection: dict[str, Expr] = {}
    for column in columns:
        builder = None
        for clause in clauses:
            value = default(column) if clause.is_delete else _value(clause, column, default)
            builder = _branch(builder, clause.condition, value)
        assert builder is not None  # non-empty `clauses` is a precondition
        projection[column] = builder.otherwise(default(column))

    keep = None
    for clause in clauses:
        keep = _branch(keep, clause.condition, lit(not clause.is_delete))
    assert keep is not None
    projection[_KEEP] = keep.otherwise(lit(default_keep))

    kept = relation.select(**projection).filter(Col(_KEEP)).select(*columns)
    return [kept]


def _value(clause: MergeClause, column: str, default: Any) -> Expr:
    """The expression a non-delete clause writes into `column`."""
    if clause.values is None:
        return Col(source_name(column))  # UPDATE SET * / INSERT * — same-named source column
    written = clause.values.get(column)
    return written if written is not None else default(column)


def _branch(builder: Any, condition: Expr | None, value: Expr) -> Any:
    """Append one ``WHEN cond THEN value`` to a `CaseBuilder` chain.

    An absent condition is an unconditional clause — ``WHEN MATCHED THEN …`` with no
    ``AND`` — which is `lit(True)`, so it becomes a branch like any other and every clause
    stays in one ordered chain.
    """
    guard = lit(True) if condition is None else condition
    return when(guard).then(value) if builder is None else builder.when(guard).then(value)
