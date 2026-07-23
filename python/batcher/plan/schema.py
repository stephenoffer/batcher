"""`SchemaRef` — a thin wrapper making `pyarrow.Schema` the source of truth.

Every plan node, contract, and operator declares its schema as a `SchemaRef`.
There is deliberately no parallel type system: Arrow types ARE Batcher's types,
so a column dtype, a cross-FFI batch, and a Rust kernel all agree by
construction. Schema compatibility is validated at plan-build time (fail fast,
before any work is scheduled).
"""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from batcher._internal.errors import did_you_mean

__all__ = ["SchemaRef", "placeholder_schema", "suggest_columns"]


def placeholder_schema(names: list[str]) -> pa.Schema:
    """Null-typed placeholders carrying just `names` — the last-resort empty-result schema.

    Used only when a zero-batch result's types cannot be inferred (an opaque `map_batches`
    output). Prefer `plan.logical.empty_result_schema`, which falls back to this.

    Args:
        names: The column names the result is expected to carry.

    Returns:
        A schema with one null-typed field per name.
    """
    return pa.schema([pa.field(name, pa.null()) for name in names])


def suggest_columns(name: str, available: list[str]) -> str:
    """A '; did you mean ...' hint for an unknown column, or '' when nothing is close.

    Turns a bare 'no column x' into an actionable message (the Postgres / Polars
    affordance). Returns up to three suggestions in this clause-shaped form, which is
    what its six call sites concatenate onto an existing sentence.

    The matching itself is `batcher._internal.errors.did_you_mean` — deliberately not a
    second `difflib` call here. That helper ranks a case-only difference first and also
    catches the abbreviation case (``"cust"`` for ``"customer_id"``) that a bare
    similarity ratio scores below any usable cutoff, and keeping one implementation is
    what stops column suggestions and format/option suggestions from drifting apart.

    Args:
        name: The column name the user supplied.
        available: The column names that do exist.

    Returns:
        ``"; did you mean 'x' or 'y'?"``, or ``""`` when nothing is close enough.

    Examples:
        .. doctest::

            >>> from batcher.plan.schema import suggest_columns
            >>> suggest_columns("nmae", ["name", "age"])
            "; did you mean 'name'?"
            >>> suggest_columns("zzz", ["name", "age"])
            ''
    """
    matches = did_you_mean(name, available)
    if not matches:
        return ""
    return f"; did you mean {' or '.join(repr(n) for n in matches)}?"


@dataclass(frozen=True, slots=True)
class SchemaRef:
    """An immutable handle to a `pyarrow.Schema`."""

    arrow: pa.Schema

    @property
    def names(self) -> list[str]:
        return list(self.arrow.names)

    def field(self, name: str) -> pa.Field:
        idx = self.arrow.get_field_index(name)
        if idx < 0:
            hint = suggest_columns(name, self.names)
            raise KeyError(f"no column named {name!r} in schema {self.names}{hint}")
        return self.arrow.field(idx)

    def has(self, name: str) -> bool:
        return self.arrow.get_field_index(name) >= 0

    @classmethod
    def from_arrow(cls, schema: pa.Schema) -> SchemaRef:
        return cls(schema)
