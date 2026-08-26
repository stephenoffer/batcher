"""What makes two set-operation branches compatible.

`Union` carries UNION, INTERSECT and EXCEPT, and this is the half of its validation that
looks at column *types* rather than names. Split out of `relational.py` for size
(`just lint-structure`); it is one concept and has no other caller.
"""

from __future__ import annotations

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher.plan.schema import SchemaRef
from batcher.plan.types import promote

__all__ = ["validate_branch_types"]


def _is_nested(dt: pa.DataType) -> bool:
    """Whether `dt` has a child type, so the flat `promote` lattice cannot judge it."""
    return (
        pa.types.is_list(dt)
        or pa.types.is_large_list(dt)
        or pa.types.is_fixed_size_list(dt)
        or pa.types.is_struct(dt)
        or pa.types.is_map(dt)
        or pa.types.is_union(dt)
    )


def validate_branch_types(schemas: list[SchemaRef | None], cols: list[str]) -> None:
    """Reject branches whose column *types* have no common supertype, at build time.

    The column names and arity are checked by the caller and raise a `PlanError` there; a
    type mismatch was the one shape that got through, and it surfaced from the engine
    mid-query as a bare `RuntimeError` ("set operation ... has incompatible branch types
    Int64 and Utf8"). The same mistake, two very different reports, one of them after the
    scan had already run.

    Only *scalar* pairs are judged. `promote` is the flat mirror of the engine's
    `common_supertype` and is not nested-aware, so it answers `None` for `list<int32>`
    against `list<int64>` -- which the engine unions perfectly well. Raising on that would
    break working queries, so a nested type on either side is left to the engine, exactly
    as before. The nested-aware supertype lives in `io.schema.evolution`, which `plan` sits
    below and must not import.

    Args:
        schemas: Each branch's statically-known schema, or `None` where it is not known.
        cols: The shared column names, already validated to be identical across branches.

    Raises:
        PlanError: If two branches give a column types with no common supertype.
    """
    if any(s is None for s in schemas):
        return  # a branch whose schema is not known statically: nothing to judge
    base = schemas[0]
    for other in schemas[1:]:
        for name in cols:
            left, right = base.field(name).type, other.field(name).type
            if _is_nested(left) or _is_nested(right):
                continue
            if promote(left, right) is None:
                raise PlanError(
                    f"union inputs disagree on the type of column {name!r}: "
                    f"{left} vs {right}, and there is no type both widen to. "
                    f"Cast one branch, e.g. .with_columns({name}=col({name!r})"
                    f".cast('{right}'))."
                )
