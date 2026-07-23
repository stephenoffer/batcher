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
from batcher.plan.types import promote

__all__ = [
    "SchemaDrift",
    "normalize_batch",
    "reconcile_batches",
    "schema_drift",
    "unify_schemas",
]


def _common_supertype(a: pa.DataType, b: pa.DataType) -> pa.DataType | None:
    """The common non-lossy supertype of `a` and `b`, recursing into nested types.

    Delegates scalar types to the neutral `plan.types.promote` lattice, then extends
    it *structurally*: a list/struct/map whose leaves each have a common supertype has
    one too. So ``list<int32>`` and ``list<int64>`` unify to ``list<int64>`` (the exact
    nested analogue of the flat ``int32``/``int64`` widening the lattice already does),
    and ``struct<a>`` merging with ``struct<a, b>`` yields ``struct<a, b>`` with ``b``
    read as null where absent — the routine "a nested field was added / its element
    width grew across files" evolution, which otherwise raised even though a lossless
    common type exists. Returns ``None`` when there is genuinely no non-lossy common
    type (an int/string collision, mismatched list kinds), so the caller still raises.
    """
    scalar = promote(a, b)
    if scalar is not None:
        return scalar
    # A `dictionary<T>` is an *encoding* of its value type `T`, not a distinct logical type:
    # a dict-encoded file and a plain-`T` file both read as `T` (DuckDB decodes both to the
    # value type). Unwrap either side and unify the values — so Parquet's routine mix of
    # dictionary and plain pages across files reconciles instead of raising.
    if pa.types.is_dictionary(a):
        return _common_supertype(a.value_type, b)
    if pa.types.is_dictionary(b):
        return _common_supertype(a, b.value_type)
    # Same logical type, wider offsets: `string`/`large_string` and `binary`/`large_binary`
    # unify to the large variant (lossless), exactly as `int32`/`int64` widen.
    if (pa.types.is_string(a) and pa.types.is_large_string(b)) or (
        pa.types.is_large_string(a) and pa.types.is_string(b)
    ):
        return pa.large_string()
    if (pa.types.is_binary(a) and pa.types.is_large_binary(b)) or (
        pa.types.is_large_binary(a) and pa.types.is_binary(b)
    ):
        return pa.large_binary()
    # Same instant type, different resolution: widen a timestamp to the finer unit when the
    # timezone matches. A differing timezone is a genuine semantic conflict (different
    # instants), so it still returns None and the caller raises.
    if pa.types.is_timestamp(a) and pa.types.is_timestamp(b) and a.tz == b.tz:
        order = ["s", "ms", "us", "ns"]
        finer = a.unit if order.index(a.unit) >= order.index(b.unit) else b.unit
        return pa.timestamp(finer, a.tz)
    if pa.types.is_struct(a) and pa.types.is_struct(b):
        return _merge_structs(a, b)
    if pa.types.is_map(a) and pa.types.is_map(b):
        key = _common_supertype(a.key_type, b.key_type)
        item = _common_supertype(a.item_type, b.item_type)
        if key is None or item is None:
            return None
        return pa.map_(key, item)
    a_kind = _list_kind(a)
    if a_kind is not None and a_kind == _list_kind(b):
        value = _common_supertype(a.value_type, b.value_type)
        if value is None:
            return None
        # A fixed-size list keeps its width only when both sides agree on it; otherwise
        # it widens to a variable list (the lengths differ across files).
        if a_kind == "fixed" and a.list_size == b.list_size:
            return pa.list_(pa.field("item", value), a.list_size)
        make = pa.large_list if a_kind == "large" else pa.list_
        return make(pa.field("item", value))
    return None


def _list_kind(t: pa.DataType) -> str | None:
    """``"list"``/``"large"``/``"fixed"`` for a list-like Arrow type, else ``None``."""
    if pa.types.is_large_list(t):
        return "large"
    if pa.types.is_fixed_size_list(t):
        return "fixed"
    if pa.types.is_list(t):
        return "list"
    return None


def _merge_structs(a: pa.DataType, b: pa.DataType) -> pa.DataType | None:
    """Union two struct types field-by-field (first-seen order), else ``None``.

    A field present in both must have a common supertype; a field on one side only is
    carried through as nullable (it reads null in the files that lack it).
    """
    b_fields = {f.name: f.type for f in b}
    fields: list[pa.Field] = []
    for f in a:
        if f.name in b_fields:
            common = _common_supertype(f.type, b_fields[f.name])
            if common is None:
                return None
            fields.append(pa.field(f.name, common))
        else:
            fields.append(pa.field(f.name, f.type))
    a_names = {f.name for f in a}
    fields.extend(pa.field(f.name, f.type) for f in b if f.name not in a_names)
    return pa.struct(fields)


def _promote(a: pa.DataType, b: pa.DataType, *, column: str) -> pa.DataType:
    """The common supertype of `a` and `b`, raising a column-named `SchemaError`.

    Wraps `_common_supertype` (the scalar `plan.types.promote` lattice extended to
    recurse through list/struct/map leaves) — turning its ``None`` (no non-lossy common
    type) into the actionable cross-file error this module raises.
    """
    common = _common_supertype(a, b)
    if common is None:
        raise SchemaError(
            f"column {column!r} has incompatible types across files: {a} vs {b} "
            "(no non-lossy common type). Cast explicitly or use schema_mode='latest'."
        )
    return common


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


def reconcile_batches(batches: list[pa.RecordBatch]) -> list[pa.RecordBatch]:
    """Reconcile a list of batches with possibly-differing schemas to one union schema.

    A no-op (fast path) when every batch already shares the first's schema. Otherwise the
    columns are unioned (missing columns become typed nulls, promotable types widened) so the
    batches can be concatenated, iterated, or written as one table. This is what lets a
    `map_batches` UDF whose output schema DRIFTS across batches — e.g. LLM structured outputs
    where later batches carry extra fields — succeed instead of failing at the concat, the
    schema-inference footgun Ray Data hits (it infers from the first batch and the merge
    fails). Vectorized Arrow kernels only; never iterates rows."""
    if len(batches) <= 1:
        return batches
    first = batches[0].schema
    if all(b.schema.equals(first) for b in batches[1:]):
        return batches
    target = unify_schemas([b.schema for b in batches], mode="union")
    return [normalize_batch(b, target) for b in batches]


def normalize_batch(batch: pa.RecordBatch, target: pa.Schema) -> pa.RecordBatch:
    """Reshape `batch` to `target`: add missing columns as typed nulls, cast
    promotable columns, and reorder to the target field order. Vectorized (no row
    iteration)."""
    import pyarrow.compute as pc

    cols: list[pa.Array] = []
    for field in target:
        if field.name in batch.schema.names:
            arr = batch.column(field.name)
            if arr.type.equals(field.type):
                cols.append(arr)
            else:
                # `promote` picks `float64` for an int/float column mix — its one
                # deliberately-lossy widening (a column stored as int in older files,
                # float in newer ones). A *safe* cast rejects any int64 above 2^53 that
                # float64 cannot hold exactly, so it would raise on exactly the data the
                # lattice already decided to coerce. DuckDB coerces such a union to
                # double (large ints rounded to the nearest float); match it. Every other
                # promotion the lattice makes is lossless and stays a safe cast.
                lossy_int_to_float = pa.types.is_integer(arr.type) and pa.types.is_floating(
                    field.type
                )
                cols.append(pc.cast(arr, field.type, safe=not lossy_int_to_float))
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
