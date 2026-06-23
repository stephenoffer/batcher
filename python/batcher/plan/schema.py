"""`SchemaRef` — a thin wrapper making `pyarrow.Schema` the source of truth.

Every plan node, contract, and operator declares its schema as a `SchemaRef`.
There is deliberately no parallel type system: Arrow types ARE Batcher's types,
so a column dtype, a cross-FFI batch, and a Rust kernel all agree by
construction. Schema compatibility is validated at plan-build time (fail fast,
before any work is scheduled).
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

import pyarrow as pa

__all__ = ["SchemaRef", "suggest_columns"]


def suggest_columns(name: str, available: list[str]) -> str:
    """A '; did you mean ...' hint for an unknown column, or '' when nothing is close.

    Turns a bare 'no column x' into an actionable message (the Postgres / Polars
    affordance). Matches case-insensitively; returns up to three suggestions.
    """
    lowered = {c.lower(): c for c in available}
    close = difflib.get_close_matches(name.lower(), list(lowered), n=3, cutoff=0.6)
    if not close:
        return ""
    names = [lowered[c] for c in close]
    return f"; did you mean {' or '.join(repr(n) for n in names)}?"


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
