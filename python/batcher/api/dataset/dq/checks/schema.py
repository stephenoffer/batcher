"""Schema constraints, answered before anything executes.

The schema is known at plan-build time, so these constraints carry their verdict rather
than an expression. That is worth doing for its own sake — a contract that costs nothing
gets run on every table — but the real reason is ordering: when a column is missing, every
value constraint written against it fails, and the report then names five broken checks
instead of the one missing column that caused them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher.api.dataset.dq.constraints import SchemaConstraint
from batcher.plan.types.registry import dtype_name, resolve_dtype

__all__ = ["column_types", "has_columns", "no_unexpected_columns"]


def _resolve(expected: Any, column: str) -> pa.DataType:
    """The expected Arrow type, whether it was spelled as a name or given as a type."""
    if isinstance(expected, pa.DataType):
        return expected
    # `resolve_dtype` returns None for a name it does not know rather than raising, and a
    # None target would reach `DataType.equals` as a TypeError from deep inside pyarrow —
    # a stack trace naming neither the column nor the type that was misspelled.
    resolved = resolve_dtype(str(expected))
    if resolved is None:
        raise PlanError(
            f"column_types({column!r}): {expected!r} is not a type name Batcher can resolve. "
            "Use a cast name such as 'int64', 'string', 'timestamp(us)', or a pyarrow type."
        )
    return resolved


def has_columns(schema: pa.Schema, names: Iterable[str]) -> SchemaConstraint:
    """Every column in `names` must be present in the schema.

    Args:
        schema: The dataset's schema.
        names: The columns the contract requires.

    Returns:
        The schema constraint, already decided.
    """
    required = list(names)
    if not required:
        raise PlanError("has_columns() requires at least one column name")
    missing = [c for c in required if c not in schema.names]
    detail = "" if not missing else f"missing: {', '.join(missing)}"
    return SchemaConstraint(f"has_columns({', '.join(required)})", not missing, detail)


def no_unexpected_columns(schema: pa.Schema, allowed: Iterable[str]) -> SchemaConstraint:
    """The schema must contain no column outside `allowed`.

    The half of a schema contract that catches a *widening* upstream change. A new column
    is harmless to every query that names its columns, right up to the point where it
    carries data nobody has classified, and then it is a governance incident.

    Args:
        schema: The dataset's schema.
        allowed: The complete set of columns the contract permits.

    Returns:
        The schema constraint, already decided.
    """
    permitted = set(allowed)
    if not permitted:
        raise PlanError("no_unexpected_columns() requires at least one allowed column name")
    extra = [c for c in schema.names if c not in permitted]
    detail = "" if not extra else f"unexpected: {', '.join(extra)}"
    return SchemaConstraint(f"no_unexpected_columns({len(permitted)} allowed)", not extra, detail)


def column_types(schema: pa.Schema, expected: Mapping[str, Any]) -> SchemaConstraint:
    """Each named column must have exactly the given type.

    A missing column counts as a type violation, so this constraint stands alone; pair it
    with `has_columns` when you want the two reported separately.

    Args:
        schema: The dataset's schema.
        expected: Column name to expected type, as a cast name or a pyarrow type.

    Returns:
        The schema constraint, already decided.
    """
    if not expected:
        raise PlanError("column_types() requires at least one column")
    wrong: list[str] = []
    for column, want in expected.items():
        target = _resolve(want, column)
        if column not in schema.names:
            wrong.append(f"{column}: missing (expected {target})")
            continue
        actual = schema.field(column).type
        if not actual.equals(target):
            wrong.append(
                f"{column}: {dtype_name(actual) or actual} != {dtype_name(target) or target}"
            )
    detail = "; ".join(wrong)
    return SchemaConstraint(f"column_types({', '.join(expected)})", not wrong, detail)
