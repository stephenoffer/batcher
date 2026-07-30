"""Static per-column byte widths derived from a column's Arrow type.

A column's width is a property of its dtype, known before any row is read: an
`int64` is 8 bytes, a `date32` is 4. Only the variable-length types (string,
binary, list, struct) genuinely need a measurement, and for those this module
returns a documented prior that a *learned* width overrides as soon as one exists.

This is the cold-start floor under the cost model's byte axes (broadcast
eligibility, memory, IO). Before it, an unmeasured relation was costed at a flat
`bytes_per_row` constant regardless of its schema — so a two-`int64`-column join
key (16 B/row) and a 20-column payload were both estimated at 64 B/row. That
single constant decided broadcast eligibility, and it mis-sized narrow relations
by ~4x, forfeiting the broadcast join that a small build side should always get.

Neutral layer: imports only `pyarrow`.
"""

from __future__ import annotations

import functools

import pyarrow as pa

__all__ = ["DEFAULT_VARLEN_BYTES", "column_bytes", "schema_row_bytes"]

# Prior for a variable-length column (string/binary/list/struct) with no measured
# width. Deliberately generous — over-estimating a payload column's width is the
# safe direction for the byte axes it feeds (it can only make the cost model more
# conservative about broadcasting a wide relation). A learned `avg_byte_width`
# replaces it the first time the column is actually measured.
DEFAULT_VARLEN_BYTES = 32.0

# Arrow's offset buffers cost 4 or 8 bytes per row on top of the value bytes.
_OFFSET_BYTES = 4.0

# Elements assumed in a variable-length `list`/`large_list` with no measured width.
# A list column's width is `len × element_width`, so charging it a flat scalar prior —
# as this module did before, ignoring the value type entirely — under-predicts by the
# element size times the length. That is not a rounding error on the columns that
# matter: a `list<float64>` embedding is ~8 B per element, so the old flat 36 B/row
# stood in for kilobytes. `fixed_size_list` (the usual embedding type) needs no prior
# at all — its length is in the type — so this only covers the genuinely unknown case,
# and a learned `avg_byte_width` replaces it the moment the column is measured.
_DEFAULT_LIST_LEN = 8.0

# A union stores one type code per row alongside its value.
_UNION_TYPE_CODE_BYTES = 1.0


def _storage_type(dtype: pa.DataType) -> pa.DataType | None:
    """The physical type an extension type is laid out as, or `None` if not one.

    Every Batcher tensor column — a decoded image, an audio waveform, a video frame stack,
    a model output — is the canonical `arrow.fixed_shape_tensor` extension type, produced by
    `io/formats/ml/tensor.py`, `ml/decode/media.py`, and `core/udf/call.py`. An extension
    type is a *label* on a storage type, and none of the `pa.types.is_*` predicates below
    see through the label, so before this every one of those columns fell through to the
    `default_varlen` prior: a 224x224x3 `uint8` image was sized at 32 B/row against a true
    150,528 — a 4,704x under-estimate, in the direction that makes a memory envelope too
    small and a build side look broadcastable when replicating it would OOM every worker.

    Unwrapping to the storage type makes it exact, because a fixed-shape tensor's storage is
    a `fixed_size_list` whose length is in the type.
    """
    storage = getattr(dtype, "storage_type", None)
    return storage if isinstance(storage, pa.DataType) else None


@functools.lru_cache(maxsize=2048)
def column_bytes(dtype: pa.DataType, default_varlen: float = DEFAULT_VARLEN_BYTES) -> float:
    """Estimated bytes per row for a column of `dtype`.

    Fixed-width types are exact (the Arrow bit width). Variable-length types return
    `default_varlen` plus their offset-buffer cost, since their true width needs a
    measurement.

    Memoized on `(dtype, default_varlen)`, which is the whole input: this is a pure
    function of an Arrow type, and Arrow types are immutable and hash by value. It is worth
    caching because it is called per column per plan node per estimate, while a query has
    only a handful of distinct types — a warm TPC-H q8 made **1,109 calls covering 15
    distinct types**, and the walk is not cheap (a chain of `pa.types.is_*` predicates, plus
    recursion for nested types, where a `struct` or `map` re-derives every field's width on
    every call). The recursive arms below re-enter through this same wrapper, so a nested
    type's children are cached too.

    Args:
        dtype: The column's Arrow type.
        default_varlen: Bytes to assume for a variable-length value.

    Returns:
        The estimated per-row width of the column, in bytes.
    """
    storage = _storage_type(dtype)
    if storage is not None:
        # An extension type is a label on a storage layout; the bytes are the storage's.
        return column_bytes(storage, default_varlen)
    if pa.types.is_null(dtype):
        # An all-null column has no value buffer at all — Arrow's `null` type is pure
        # metadata. Charging it the variable-length prior (the previous fall-through)
        # invented 32 B/row for a column that occupies none, which inflates the width of
        # every relation carrying a not-yet-typed column: the shape a JSON or CSV source
        # produces for a field it saw only nulls in, and a schema-evolution placeholder.
        return 0.0
    if pa.types.is_boolean(dtype):
        return 1.0
    if pa.types.is_dictionary(dtype):
        # The values live in a shared dictionary; a row costs only its index.
        return float(max(dtype.index_type.bit_width // 8, 1))
    if pa.types.is_string(dtype) or pa.types.is_binary(dtype):
        return default_varlen + _OFFSET_BYTES
    if pa.types.is_large_string(dtype) or pa.types.is_large_binary(dtype):
        return default_varlen + 2 * _OFFSET_BYTES
    if _is_string_view(dtype):
        # A view type replaces the offset buffer with a 16-byte view struct that inlines
        # short values; long ones still point into a data buffer. So the per-row cost is the
        # view plus the value, not the value plus an offset.
        return default_varlen + 16.0
    if pa.types.is_fixed_size_list(dtype):
        # Exact: the length is in the type, so no measurement is needed and no offset
        # buffer exists. This is the embedding/vector column, and it is the one nested
        # case whose width is fully known statically.
        return dtype.list_size * column_bytes(dtype.value_type, default_varlen)
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype) or _is_list_view(dtype):
        # A plain `list` carries one 32-bit offset per row and a `large_list` one 64-bit
        # offset. A *view* list carries an offset **and** a size, so its per-row bookkeeping
        # is two buffers wide either way — which is what `2 * _OFFSET_BYTES` charges here.
        offsets = _OFFSET_BYTES if pa.types.is_list(dtype) else 2 * _OFFSET_BYTES
        return _DEFAULT_LIST_LEN * column_bytes(dtype.value_type, default_varlen) + offsets
    if pa.types.is_map(dtype):
        # A map is a list of key/value entries and is how every semi-structured source
        # (JSON objects with open-ended keys, Parquet `MAP` groups, Avro maps) lands in
        # Arrow. It matches none of the list predicates above, so it used to fall through
        # to the flat scalar prior — a `map<string, string>` of eight entries was sized at
        # 32 B/row against a true ~600, which is the same under-count as the tensor case on
        # exactly the data type that carries a document's payload.
        entry = column_bytes(dtype.key_type, default_varlen) + column_bytes(
            dtype.item_type, default_varlen
        )
        return _DEFAULT_LIST_LEN * entry + _OFFSET_BYTES
    if pa.types.is_struct(dtype):
        return sum(column_bytes(f.type, default_varlen) for f in dtype)
    if _is_run_end_encoded(dtype):
        # Run-end encoding stores one run-end index and one value per *run*. How many runs
        # a column has is a measurement, not a property of the type, so the width is charged
        # at the worst case of one run per row — an honest upper bound rather than an
        # invented compression ratio, and still far below the 32 B prior this used to hit
        # for a `run_end_encoded<int32, int64>` whose worst case is 12.
        return column_bytes(dtype.run_end_type, default_varlen) + column_bytes(
            dtype.value_type, default_varlen
        )
    if pa.types.is_union(dtype):
        return _union_bytes(dtype, default_varlen)
    try:
        bit_width = dtype.bit_width
    except (ValueError, NotImplementedError, AttributeError):
        # A type with no fixed bit width and no branch above.
        return default_varlen
    return float(max(bit_width // 8, 1))


def _union_bytes(dtype: pa.DataType, default_varlen: float) -> float:
    """Bytes per row of a union column, by its two physical layouts.

    A **sparse** union allocates a full child array for every variant, so a row occupies
    the sum of all of them plus its type code. A **dense** union allocates only the chosen
    variant, so a row costs the average variant plus a type code and an offset. Both used
    to fall through to the scalar prior, which under-reads a sparse union of five payload
    variants by roughly its arity — the shape an Avro union or a polymorphic JSON field
    takes.
    """
    children = [column_bytes(dtype.field(i).type, default_varlen) for i in range(dtype.num_fields)]
    if not children:
        return _UNION_TYPE_CODE_BYTES
    # `pyarrow` exposes only `is_union`; the layout is on the type's `mode`.
    if getattr(dtype, "mode", "dense") == "sparse":
        return sum(children) + _UNION_TYPE_CODE_BYTES
    return sum(children) / len(children) + _UNION_TYPE_CODE_BYTES + _OFFSET_BYTES


def _is_string_view(dtype: pa.DataType) -> bool:
    """Whether `dtype` is a `string_view`/`binary_view`, on a `pyarrow` that has them."""
    return _has_type(dtype, "is_string_view", "is_binary_view")


def _is_list_view(dtype: pa.DataType) -> bool:
    """Whether `dtype` is a `list_view`/`large_list_view`, on a `pyarrow` that has them."""
    return _has_type(dtype, "is_list_view", "is_large_list_view")


def _is_run_end_encoded(dtype: pa.DataType) -> bool:
    """Whether `dtype` is run-end encoded, on a `pyarrow` that has the predicate."""
    return _has_type(dtype, "is_run_end_encoded")


def _has_type(dtype: pa.DataType, *predicates: str) -> bool:
    """Whether any named `pa.types` predicate exists and matches `dtype`.

    The view and run-end-encoded predicates arrived in different `pyarrow` releases, and
    this module is the neutral floor under every byte axis — it must not raise on an older
    interpreter just because a type it can now size does not exist there.
    """
    for name in predicates:
        predicate = getattr(pa.types, name, None)
        if predicate is not None and predicate(dtype):
            return True
    return False


@functools.lru_cache(maxsize=1024)
def schema_row_bytes(schema: pa.Schema, default_varlen: float = DEFAULT_VARLEN_BYTES) -> float:
    """Estimated bytes per row for a whole schema — the sum of its columns' widths.

    Args:
        schema: The Arrow schema to size.
        default_varlen: Bytes to assume for a variable-length value.

    Returns:
        The estimated per-row width of one row of `schema`, in bytes.
    """
    return sum(column_bytes(f.type, default_varlen) for f in schema)
