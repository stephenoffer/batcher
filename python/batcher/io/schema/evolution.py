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
    "conform_batch",
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
    # The scalar arms that used to sit here — `string`/`large_string` offset widening and
    # timestamp resolution — are the shared `promote` lattice's job and were restated in
    # this file. A second answer to the same question in a second place is exactly how the
    # io layer and the engine came to disagree about a `timestamp[ms]` file. What remains
    # is genuinely this module's own: the *structural* recursion into nested types, which
    # only the file-level reader needs. (`promote` unwraps a dictionary of a scalar; the
    # arms below re-do it so a `dictionary<list<...>>` reaches the nested recursion too.)
    if pa.types.is_dictionary(a):
        return _common_supertype(a.value_type, b)
    if pa.types.is_dictionary(b):
        return _common_supertype(a, b.value_type)
    tensor = _tensor_common(a, b)
    if tensor is not None:
        return tensor
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


def _is_ragged(dtype: pa.DataType) -> bool:
    """Whether `dtype` is the ragged-tensor struct, without importing at module scope."""
    from batcher.io.formats.ml.ragged import is_ragged_tensor_column

    return is_ragged_tensor_column(dtype)


def _tensor_common(a: pa.DataType, b: pa.DataType) -> pa.DataType | None:
    """Two tensor columns that disagree on shape unify to the *ragged* encoding.

    A `map_batches` UDF that returns arrays of differing shapes produces a ragged column
    single-node, because one call sees every row and the shapes plainly differ. Distributed,
    the same UDF runs per partition -- and a partition whose rows happen to agree yields a
    `fixed_shape_tensor` of *that* shape. Two such partitions then have no common type and the
    reconcile raised, so a ragged column was a `SchemaError` on the distributed path and a
    result single-node. Which partitions agree depends on the partition count, so the same
    query failed or succeeded according to how it was scheduled.

    `struct<data, shape, dtype>` is the common type these actually have: it is what the
    single-node path already picks for the same rows, so unifying to it makes the distributed
    answer the single-node answer rather than inventing a third one.

    Returns `None` for anything that is not a pair of tensor columns, so every other type
    keeps the behaviour it had -- including two *identical* fixed-shape tensors, which unify
    to themselves and must not be widened to ragged.
    """
    from batcher.io.formats.ml.ragged import is_ragged_tensor_column, ragged_tensor_type

    def tensorish(t: pa.DataType) -> bool:
        return isinstance(t, pa.FixedShapeTensorType) or is_ragged_tensor_column(t)

    if not (tensorish(a) and tensorish(b)):
        return None
    return a if a.equals(b) else ragged_tensor_type()


def _fixed_tensor_to_ragged(array: pa.Array) -> pa.Array:
    """A fixed-shape-tensor column re-encoded as a ragged one, row shapes preserved.

    The cast `normalize_batch` would otherwise reach for cannot do this: `struct` is not a
    cast target for an extension type, and the row buffers have to be rewritten anyway.
    """
    from batcher.io.formats.ml.ragged import to_ragged_tensor_column

    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    if array.null_count:
        # `to_numpy_ndarray` refuses a column with nulls, so the null rows are carried
        # through as `None` -- which `to_ragged_tensor_column` stores as a null row.
        valid = array.is_valid().to_pylist()
        dense = array.drop_null().to_numpy_ndarray()
        rows: list[object] = []
        it = iter(dense)
        rows = [next(it) if ok else None for ok in valid]
        return to_ragged_tensor_column(rows)
    return to_ragged_tensor_column(list(array.to_numpy_ndarray()))


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
    iteration).

    The source column index is built **once** per batch rather than being re-derived per
    target field. `Schema.names` is a property that materializes a fresh Python list on
    every access, so asking `field.name in batch.schema.names` inside the loop was
    quadratic in the column count — and this runs per batch on every schema-evolving read
    (single-node and on each distributed worker via `NormalizedFileSplit`) and on every
    drifting `map_batches` output via `reconcile_batches`. Measured cost of the lookup
    alone, per batch: 16 columns 0.10 ms, 64 columns 1.4 ms, 256 columns 25 ms, 1,024
    columns 323 ms. A wide table did not read slowly so much as stop reading: the control
    plane spent a third of a second per 16,384-row morsel deciding which columns it had.
    A dict keyed on the names list makes it linear (1,024 columns: 323 ms -> 1.9 ms).
    """
    import pyarrow.compute as pc

    names = batch.schema.names
    source = {name: i for i, name in enumerate(names)}
    # A duplicate column name makes the index ambiguous, and `RecordBatch.column(name)`
    # raises on exactly that rather than picking one. Keeping the raise is the point: the
    # dict would silently resolve to whichever occurrence it kept, so an ambiguous batch
    # would start returning *a* column instead of saying it cannot choose.
    ambiguous = len(source) != len(names)
    cols: list[pa.Array] = []
    for field in target:
        index = source.get(field.name)
        if index is not None:
            arr = batch.column(field.name) if ambiguous else batch.column(index)
            if arr.type.equals(field.type):
                cols.append(arr)
            elif isinstance(arr.type, pa.FixedShapeTensorType) and _is_ragged(field.type):
                # The one unification whose target is not reachable by a cast; see
                # `_tensor_common` for why two partitions can disagree in the first place.
                cols.append(_fixed_tensor_to_ragged(arr))
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


def conform_batch(batch: pa.RecordBatch, target: pa.Schema, *, path: str) -> pa.RecordBatch:
    """Reshape `batch` to `target` under ``schema_mode='strict'``, or raise `SchemaError`.

    Strict mode declares that the **first file's schema is the contract** for the whole
    source, and everything downstream — the plan's output schema, the optimizer's column
    pruning, the final `Table.from_batches` — trusts that declaration. Nothing was
    enforcing it, so a file that disagreed produced one of two silent-ish failures: an
    extra column was dropped without a word, and a differing *type* surfaced hundreds of
    lines later as a bare `pyarrow.lib.ArrowInvalid` from the concat, after the whole
    read had been paid for. This makes the declaration true at the point it is made.

    The rules match DuckDB's non-``union_by_name`` reader, which has the same "file 0
    defines the contract" semantics:

    - a column in the file but not in `target` is **dropped** (it is not in the contract);
    - a column in `target` but not in the file **raises** — the contract promised it;
    - a differing type is **cast** to the contract's type.

    The cast is *safe*, which is the one deliberate departure from DuckDB: DuckDB reads a
    ``float64`` file against an ``int64`` contract by truncating, so ``2.5`` silently
    becomes ``2``. Silently corrupting a value is exactly the failure mode this function
    exists to remove, so a lossy conformance raises and names ``schema_mode='union'``,
    which promotes the column to ``float64`` and returns ``2.5`` intact.

    Args:
        batch: One batch as the file produced it.
        target: The source's declared schema (already narrowed to any projection).
        path: The file `batch` came from, for the error message.

    Returns:
        `batch` reshaped to `target`, or `batch` itself when it already conforms.

    Raises:
        SchemaError: If `path` lacks a declared column, or holds one whose type cannot be
            cast to the declared type without losing data.
    """
    import pyarrow.compute as pc

    if batch.schema.equals(target):
        return batch
    present = set(batch.schema.names)
    cols: list[pa.Array] = []
    for field in target:
        if field.name not in present:
            raise SchemaError(
                f"file {path!r} is missing column {field.name!r}, which the source's schema "
                f"declares (it is present in the first file). Columns here: "
                f"{sorted(present)}. Use schema_mode='union' to read files whose columns "
                "differ, filling the absent ones with nulls."
            )
        arr = batch.column(field.name)
        if arr.type.equals(field.type):
            cols.append(arr)
            continue
        try:
            cols.append(pc.cast(arr, field.type))
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
            raise SchemaError(
                f"file {path!r} has column {field.name!r} as {arr.type}, but the source's "
                f"schema declares {field.type} (from the first file), and the values do not "
                "convert without loss. Use schema_mode='union' to promote the column to a "
                "type that holds both files."
            ) from exc
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
