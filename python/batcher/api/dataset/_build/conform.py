"""Bodies of the `Dataset` verbs that hold a relation to a shape: `match_to_schema`, `drop_nans`.

Both are decided from the schema alone and lower to a `select` or a `filter`, so neither
reads a row at plan time or adds IR.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir import Col, Expr, coalesce, lit, null
from batcher.plan.types import dtype_name, normalize_dtype_spec, resolve_dtype, widen

if TYPE_CHECKING:
    from batcher.api.dataset.frame import Dataset

__all__ = ["build_drop_nans", "build_match_to_schema"]

_MISSING = ("raise", "insert")
_EXTRA = ("raise", "ignore")


def _target_type(column: str, spec: Any) -> pa.DataType:
    """The Arrow type a schema entry names, widened the way the engine boundary widens it."""
    name = normalize_dtype_spec(spec, caller="match_to_schema")
    resolved = resolve_dtype(name)
    if resolved is None:
        raise PlanError(f"match_to_schema(): column {column!r} names unknown dtype {spec!r}")
    return widen(resolved)


def _missing_value(column: str, policy: str | Expr, dtype: pa.DataType) -> Expr:
    """The expression filling a schema column the input lacks, or a refusal."""
    if isinstance(policy, Expr):
        return policy.cast(_named(column, dtype))
    if policy not in _MISSING:
        raise PlanError(
            f"match_to_schema(): missing_columns for {column!r} must be one of "
            f"{list(_MISSING)} or an expression, got {policy!r}"
        )
    if policy == "raise":
        raise PlanError(
            f"match_to_schema(): column {column!r} is in the schema but not in the input; pass "
            "missing_columns='insert' to add it as nulls"
        )
    return null().cast(_named(column, dtype))


def _named(column: str, dtype: pa.DataType) -> str:
    """The cast-target name for `dtype`, refusing a type the cast grammar cannot spell."""
    name = dtype_name(dtype)
    if name is None:
        raise PlanError(
            f"match_to_schema(): cannot build column {column!r} of type {dtype}; the cast "
            "vocabulary has no name for it"
        )
    return name


def build_match_to_schema(
    ds: Dataset,
    schema: Mapping[str, Any] | pa.Schema,
    missing_columns: str | Mapping[str, str | Expr],
    extra_columns: str,
) -> Dataset:
    """Project `ds` onto `schema`'s columns, in its order, checking types (see the method).

    Args:
        ds: The relation to conform.
        schema: Column name to dtype, or a pyarrow schema.
        missing_columns: ``"raise"``, ``"insert"``, or a per-column mapping of either or of
            an expression computing the column.
        extra_columns: ``"raise"`` or ``"ignore"`` for input columns the schema lacks.

    Returns:
        A relation with exactly `schema`'s columns.

    Raises:
        PlanError: On a type mismatch, a column the policies refuse, or a bad policy.
    """
    if extra_columns not in _EXTRA:
        raise PlanError(
            f"match_to_schema(): extra_columns must be one of {list(_EXTRA)}, got {extra_columns!r}"
        )
    wanted = {f.name: f.type for f in schema} if isinstance(schema, pa.Schema) else dict(schema)
    extra = [c for c in ds.columns if c not in wanted]
    if extra and extra_columns == "raise":
        raise PlanError(
            f"match_to_schema(): input column(s) {extra} are not in the schema; pass "
            "extra_columns='ignore' to drop them"
        )
    have = ds.schema
    items: list[Expr] = []
    for column, spec in wanted.items():
        target = _target_type(column, spec)
        if column not in ds.columns:
            policy = (
                missing_columns.get(column, "raise")
                if isinstance(missing_columns, Mapping)
                else missing_columns
            )
            items.append(_missing_value(column, policy, target).alias(column))
            continue
        actual = have.field(column).type
        if not actual.equals(target):
            raise PlanError(
                f"match_to_schema(): column {column!r} is {actual} but the schema says "
                f"{target}; cast it first with ds.cast({{{column!r}: ...}})"
            )
        items.append(Col(column))
    return ds.select(*items)


def build_drop_nans(ds: Dataset, subset: list[str] | None) -> Dataset:
    """Drop the rows holding a NaN in any of `subset`'s float columns (see `Dataset.drop_nans`).

    Args:
        ds: The relation to filter.
        subset: The columns to check; ``None`` checks every floating-point column.

    Returns:
        `ds` without those rows. A null is not a NaN, so a row with nulls survives.

    Raises:
        PlanError: If a named column is unknown or not floating-point.
    """
    schema = ds.schema
    floats = [f.name for f in schema if pa.types.is_floating(f.type)]
    if subset is None:
        columns = floats
    else:
        unknown = [c for c in subset if c not in ds.columns]
        if unknown:
            raise PlanError(f"drop_nans(): unknown column(s) {unknown}")
        not_float = [c for c in subset if c not in set(floats)]
        if not_float:
            raise PlanError(
                f"drop_nans(): column(s) {not_float} are not floating-point, so they cannot "
                "hold NaN; use drop_nulls() for missing values"
            )
        columns = list(subset)
    if not columns:
        return ds
    # `is_nan` is null on a null input, and a null predicate drops the row; a null is not a
    # NaN, so read it as false before negating.
    has_nan = coalesce(Col(columns[0]).is_nan(), lit(False))
    for c in columns[1:]:
        has_nan = has_nan | coalesce(Col(c).is_nan(), lit(False))
    return ds.filter(~has_nan)
