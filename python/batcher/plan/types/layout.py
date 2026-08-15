"""Respell the Arrow *layouts* the FFI boundary cannot import (neutral layer).

`lattice.widen` says what type the engine produces for a column; this module makes that
true for the layouts arrow-rs's C Data Interface reader rejects outright. The two are
different questions. Widening is a *type* normalization the Rust boundary performs itself
(`bc_py::normalize_batch`), so Python only has to predict it. A layout arrow-rs cannot
import never reaches that code at all -- the import fails first, with a raw
``ArrowException: The datatype "+vl" is still not supported in Rust implementation`` -- so
the respelling has to happen on the Python side of the boundary or not at all.

Today that set is exactly the **list-view layouts** (``list_view``, ``large_list_view``),
at any nesting depth. Their sibling view layouts (``string_view``, ``binary_view``) import
fine and are normalized in Rust like any other type, so they are deliberately not here:
this module respells what *cannot cross*, never what merely changes type once across.

The cast target is taken from `lattice.widen`, never restated, so a column's post-boundary
type has one definition. That mattered here: `widen` already promised ``list_view`` arrives
as ``list``, which made `Dataset.schema` report ``list<item: int64>`` for a query whose
`collect()` raised the FFI error above.

The rebuild is hand-rolled rather than a `cast` because **pyarrow's own cast is wrong
here**: through 19.0.1 every path (`Array.cast`, `pc.cast`, `Table.cast`,
`RecordBatch.cast`) emits a `list` whose offsets buffer is one slot short, so the result
fails `validate(full=True)` and corrupts the read that follows. Everything below is bulk
Arrow compute -- `list_flatten`, `list_value_length`, `cumulative_sum` -- so no per-row
Python touches the data.

Neutral layer: imports only `pyarrow` and `plan.types`.
"""

from __future__ import annotations

import pyarrow as pa

from batcher.plan.types.lattice import widen

__all__ = ["importable_array", "importable_batch", "importable_type"]


def _is(t: pa.DataType, *predicates: str) -> bool:
    """Whether `t` matches any of `predicates`, on a `pyarrow` that defines them.

    The view predicates arrived in `pyarrow` well after the types they test for, so calling
    one unguarded turns an older runtime into an `AttributeError` at import of this module.
    """
    return any(
        check(t) for name in predicates if (check := getattr(pa.types, name, None)) is not None
    )


def _has_unimportable(dt: pa.DataType) -> bool:
    """Whether `dt` contains a layout the FFI reader rejects, at any nesting depth."""
    if _is(dt, "is_list_view", "is_large_list_view"):
        return True
    if pa.types.is_struct(dt):
        return any(_has_unimportable(f.type) for f in dt)
    if pa.types.is_list(dt) or pa.types.is_large_list(dt) or pa.types.is_fixed_size_list(dt):
        return _has_unimportable(dt.value_type)
    if pa.types.is_map(dt):
        return _has_unimportable(dt.key_type) or _has_unimportable(dt.item_type)
    if pa.types.is_dictionary(dt):
        return _has_unimportable(dt.value_type)
    return False


def importable_type(dt: pa.DataType) -> pa.DataType | None:
    """The type `dt` must become before it can cross the FFI, or None if it already can.

    Returns None for the overwhelming majority of columns, so a caller pays one schema walk
    and no data touch at all. When a respelling *is* needed the target is `widen`'s, so the
    column arrives as the type `Dataset.schema` already reported.

    Args:
        dt: The column's Arrow type.

    Returns:
        The target type, or None when `dt` is already importable as-is.
    """
    return widen(dt) if _has_unimportable(dt) else None


def importable_batch(batch: pa.RecordBatch) -> pa.RecordBatch:
    """`batch` with every FFI-unimportable column respelled; the same object when none is.

    The common case is a schema walk and an early return -- no column is touched and no
    buffer is copied.

    Args:
        batch: A record batch about to cross into the engine.

    Returns:
        A batch every column of which arrow-rs can import.
    """
    targets = {f.name: t for f in batch.schema if (t := importable_type(f.type)) is not None}
    if not targets:
        return batch
    schema = pa.schema(
        [f.with_type(targets[f.name]) if f.name in targets else f for f in batch.schema]
    )
    columns = [
        importable_array(batch.column(i), targets[f.name]) if f.name in targets else batch.column(i)
        for i, f in enumerate(batch.schema)
    ]
    return pa.RecordBatch.from_arrays(columns, schema=schema)


def importable_array(arr: pa.Array, target: pa.DataType) -> pa.Array:
    """`arr` rebuilt as `target`, respelling every list-view layout inside it.

    Only the containers on the path to a list view are rebuilt; a child with no view
    beneath it is cast normally (`cast` is correct for everything except the view layouts
    themselves) and its buffers are shared, not copied.

    Args:
        arr: The array to respell.
        target: The type it must become, from `importable_type`.

    Returns:
        An equal array whose layout arrow-rs can import.
    """
    if not _has_unimportable(arr.type):
        return arr if arr.type == target else arr.cast(target)
    if _is(arr.type, "is_list_view", "is_large_list_view"):
        return _list_from_view(arr, target)
    if pa.types.is_struct(arr.type):
        children = [
            importable_array(arr.field(i), target.field(i).type) for i in range(arr.type.num_fields)
        ]
        return pa.StructArray.from_arrays(children, fields=list(target), mask=_null_mask(arr))
    if pa.types.is_fixed_size_list(arr.type):
        # `.values`, not `.flatten()`: a fixed-size list keeps its child slots under a null
        # row, and `flatten()` drops them -- which shortens the child by `list_size` per
        # null and silently slides every later row's values up. `.values` is the *whole*
        # child of the unsliced array, so the slice has to be reapplied here.
        size = arr.type.list_size
        child = arr.values.slice(arr.offset * size, len(arr) * size)
        return pa.FixedSizeListArray.from_arrays(
            importable_array(child, target.value_type), size, mask=_null_mask(arr)
        )
    if pa.types.is_map(arr.type):
        # Rebuilt from buffers rather than through the list kernels every other container
        # here uses: `list_flatten` and `list_value_length` refuse a `map` whose value type
        # contains a list view ("has no kernel matching input types"), so there is no
        # kernel path to the entries at all. The map's own validity and offsets buffers are
        # already the layout arrow-rs wants, so only the entries child is rebuilt -- and
        # `.keys`/`.items` are the *whole* child of the unsliced array, which is exactly
        # what `offset` below is still relative to.
        entries = pa.StructArray.from_arrays(
            [
                importable_array(arr.keys, target.key_type),
                importable_array(arr.items, target.item_type),
            ],
            fields=[target.key_field, target.item_field],
        )
        return pa.Array.from_buffers(
            target,
            len(arr),
            arr.buffers()[:2],
            null_count=arr.null_count,
            offset=arr.offset,
            children=[entries],
        )
    if pa.types.is_dictionary(arr.type):
        # `widen` decodes a dictionary to its value type, which is also what the boundary
        # does, so the respelled column is the decoded one rather than a re-encoded view.
        return importable_array(arr.dictionary_decode(), target)
    # list / large_list: same shape, values respelled.
    return _list_from_lengths(arr, target)


def _null_mask(arr: pa.Array) -> pa.Array | None:
    """`arr`'s null positions as a boolean mask, or None when it has no nulls.

    None rather than an all-false mask, because a mask makes `from_arrays` attach a
    validity buffer -- and Arrow *rejects* one on a map's key child ("Map array keys array
    should have no nulls"), aborting the process rather than raising. It is also the
    cheaper answer for the common fully-valid array.
    """
    return arr.is_null(nan_is_null=False) if arr.null_count else None


def _offsets_from_lengths(arr: pa.Array) -> pa.Array:
    """Int32 offsets for `arr`'s per-row lengths, with the leading zero.

    Derived from the lengths rather than read off `arr.offsets`, which is what makes this
    correct for a *view* layout: a list view's ranges may overlap, repeat, or run backwards,
    so only `list_flatten`'s row-order concatenation and the lengths beside it describe the
    same relation the offsets-and-values layout does.
    """
    import pyarrow.compute as pc

    lengths = pc.fill_null(pc.list_value_length(arr), 0).cast(pa.int32())
    if len(lengths) == 0:
        return pa.array([0], pa.int32())
    running = pc.cumulative_sum(lengths).cast(pa.int32())
    if isinstance(running, pa.ChunkedArray):
        running = running.combine_chunks()
    return pa.concat_arrays([pa.array([0], pa.int32()), running])


def _flat_values(arr: pa.Array) -> pa.Array:
    """`arr`'s child values in row order, null rows contributing nothing."""
    import pyarrow.compute as pc

    return pc.list_flatten(arr)


def _list_from_view(arr: pa.Array, target: pa.DataType) -> pa.Array:
    """A list-view array rebuilt as the `list` its `widen` target names."""
    values = importable_array(_flat_values(arr), target.value_type)
    return pa.ListArray.from_arrays(_offsets_from_lengths(arr), values, mask=_null_mask(arr))


def _list_from_lengths(arr: pa.Array, target: pa.DataType) -> pa.Array:
    """A `list`/`large_list` array rebuilt with its values respelled.

    Rebuilt through the same lengths path as a view rather than by reusing `arr.offsets`,
    because a sliced array's offsets buffer is not relative to the slice and `from_arrays`
    would silently read the wrong rows.
    """
    values = importable_array(_flat_values(arr), target.value_type)
    make = pa.LargeListArray if pa.types.is_large_list(target) else pa.ListArray
    offsets = _offsets_from_lengths(arr)
    if make is pa.LargeListArray:
        offsets = offsets.cast(pa.int64())
    return make.from_arrays(offsets, values, mask=_null_mask(arr))
