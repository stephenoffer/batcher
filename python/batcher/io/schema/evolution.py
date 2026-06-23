"""Schema reconciliation for multi-file reads — column union, type promotion, drift.

Reading a directory of files written over time is the canonical ETL ingestion
pain: ``day=1`` has columns ``[a, b]``, ``day=30`` has ``[a, b, c]`` and a column
that was ``int`` is now ``float``. This module reconciles those into one schema and
normalizes each file's batches to it.

It is pure (functions over ``pyarrow.Schema``/``RecordBatch``) and lives in the
neutral ``io`` layer — the same character as the Rust FFI narrow-type widening
(``Int32→Int64`` at the boundary), just driven by the file-level schema and applied
one layer earlier where the heterogeneity is known. It never iterates a row: every
operation is a vectorized Arrow kernel (``cast`` / ``nulls`` / column reorder).
"""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from batcher._internal.errors import SchemaError

__all__ = ["SchemaDrift", "normalize_batch", "schema_drift", "unify_schemas"]


def _is_int(t: pa.DataType) -> bool:
    return pa.types.is_integer(t)


def _is_float(t: pa.DataType) -> bool:
    return pa.types.is_floating(t)


def _promote(a: pa.DataType, b: pa.DataType, *, column: str) -> pa.DataType:
    """The common supertype of `a` and `b` under a conservative, never-lossy lattice.

    ``null`` adopts the other side; integers widen to ``int64``; floats widen to
    ``float64``; an int/float mix promotes to ``float64``. Anything else that is not
    already equal is incompatible and raises a `SchemaError` naming the column.
    """
    if a.equals(b):
        return a
    if pa.types.is_null(a):
        return b
    if pa.types.is_null(b):
        return a
    if _is_int(a) and _is_int(b):
        return pa.int64()
    if (_is_float(a) or _is_int(a)) and (_is_float(b) or _is_int(b)):
        return pa.float64()
    raise SchemaError(
        f"column {column!r} has incompatible types across files: {a} vs {b} "
        "(no non-lossy common type). Cast explicitly or use schema_mode='latest'."
    )


def unify_schemas(schemas: list[pa.Schema], mode: str = "union") -> pa.Schema:
    """Reconcile `schemas` into one, per `mode`.

    - ``"strict"``: every schema must equal the first; any difference raises.
    - ``"union"``: the union of columns (first-seen order, then new columns
      appended), each column promoted to the common supertype of its occurrences.
    - ``"latest"``: the last schema wins; its column order and types are used (older
      files are cast toward it on read).

    Raises `SchemaError` on an incompatible type collision (``union``) or any
    mismatch (``strict``).
    """
    if not schemas:
        raise SchemaError("unify_schemas() requires at least one schema")
    if mode == "strict":
        first = schemas[0]
        for s in schemas[1:]:
            if not s.equals(first):
                raise SchemaError(
                    "schema_mode='strict' but files have differing schemas: "
                    f"{first} vs {s}. Use schema_mode='union' to reconcile them."
                )
        return first
    if mode == "latest":
        return schemas[-1]
    if mode != "union":
        raise SchemaError(f"unknown schema_mode {mode!r}; use 'strict'/'union'/'latest'")

    fields: dict[str, pa.DataType] = {}
    for s in schemas:
        for f in s:
            fields[f.name] = (
                _promote(fields[f.name], f.type, column=f.name) if f.name in fields else f.type
            )
    return pa.schema([pa.field(name, t) for name, t in fields.items()])


def normalize_batch(batch: pa.RecordBatch, target: pa.Schema) -> pa.RecordBatch:
    """Reshape `batch` to `target`: add missing columns as typed nulls, cast
    promotable columns, and reorder to the target field order. Vectorized (no row
    iteration)."""
    import pyarrow.compute as pc

    cols: list[pa.Array] = []
    for field in target:
        if field.name in batch.schema.names:
            arr = batch.column(field.name)
            cols.append(arr if arr.type.equals(field.type) else pc.cast(arr, field.type))
        else:
            cols.append(pa.nulls(batch.num_rows, type=field.type))
    return pa.RecordBatch.from_arrays(cols, schema=target)


@dataclass(frozen=True, slots=True)
class SchemaDrift:
    """How an `inferred` schema differs from an `expected` one."""

    added: tuple[str, ...]
    removed: tuple[str, ...]
    type_changed: tuple[tuple[str, str, str], ...]  # (column, expected_type, actual_type)

    @property
    def has_drift(self) -> bool:
        return bool(self.added or self.removed or self.type_changed)


def schema_drift(inferred: pa.Schema, expected: pa.Schema) -> SchemaDrift:
    """Compare an `inferred` schema against an `expected` one and report the drift —
    columns added/removed and columns whose type changed. The basis for schema-drift
    detection and alerting on a daily ingest."""
    inferred_names = set(inferred.names)
    expected_names = set(expected.names)
    added = tuple(n for n in inferred.names if n not in expected_names)
    removed = tuple(n for n in expected.names if n not in inferred_names)
    changed = tuple(
        (n, str(expected.field(n).type), str(inferred.field(n).type))
        for n in inferred.names
        if n in expected_names and not inferred.field(n).type.equals(expected.field(n).type)
    )
    return SchemaDrift(added=added, removed=removed, type_changed=changed)
