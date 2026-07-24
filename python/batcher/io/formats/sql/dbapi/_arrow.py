"""Turning DB-API rows into Arrow, faithfully.

A PEP 249 driver hands back Python objects and, for most drivers, no usable type
information at all. Everything here exists to get from that to a typed Arrow batch
without inventing a type or losing a value — which is the whole correctness surface of
the DB-API path, and is kept separate from the connection handling for that reason.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher._internal.logging import note_suppressed

__all__ = ["arrow_type", "reconcile", "rows_to_batch"]


def arrow_type(module: Any, type_code: Any) -> pa.DataType | None:
    """Map a `cursor.description` type code to an Arrow type via PEP 249's type objects.

    PEP 249 requires a driver to expose `STRING`, `BINARY`, `NUMBER`, and `DATETIME`
    singletons that compare equal to the type codes it reports. That comparison is the
    only *portable* type information a DB-API driver offers, and it is coarse: `NUMBER`
    covers integers, floats, and decimals alike, so it cannot be resolved to one Arrow
    type here.

    Returning None is therefore the common, expected outcome, and it is not a failure —
    it tells `DBAPISource.schema` to fall back to inferring from real data rather than
    to guess. Guessing `int64` for a `NUMBER` column that holds prices would corrupt
    every value silently, which is exactly the class of bug `probe_is_typed` exists to
    prevent elsewhere in this package.

    Args:
        module: The driver module, which carries the PEP 249 type singletons.
        type_code: The second element of a `cursor.description` entry.

    Returns:
        The Arrow type, or None when it cannot be determined portably.
    """
    if type_code is None:
        return None
    try:
        if type_code == getattr(module, "STRING", object()):
            return pa.string()
        if type_code == getattr(module, "BINARY", object()):
            return pa.binary()
        if type_code == getattr(module, "DATETIME", object()):
            return pa.timestamp("us")
    except Exception as exc:  # pragma: no cover - a driver whose type codes reject ==
        note_suppressed("io", "map dbapi type code", exc)
        return None
    return None


def rows_to_batch(rows: list[tuple], names: list[str], schema: pa.Schema | None) -> pa.RecordBatch:
    """Transpose a block of DB-API rows into one Arrow `RecordBatch`.

    `zip(*rows)` is the whole row-to-column step, and it runs once per *batch*, not once
    per row — this is the single place row-shaped data becomes columnar.

    When `schema` is known the batch is built against it so every batch of a multi-batch
    read has identical types; without it Arrow infers, which is only safe for the first
    batch (the caller then reuses that inferred schema for the rest).
    """
    if not rows:
        if schema is not None:
            return pa.RecordBatch.from_pylist([], schema=schema)
        return pa.RecordBatch.from_pylist([])
    columns = [list(col) for col in zip(*rows, strict=False)]
    if schema is not None:
        arrays = [pa.array(col, type=schema.field(i).type) for i, col in enumerate(columns)]
        return pa.RecordBatch.from_arrays(arrays, schema=schema)
    return pa.RecordBatch.from_arrays([pa.array(col) for col in columns], names=names)


def reconcile(running: pa.Schema | None, batch: pa.RecordBatch) -> tuple[pa.Schema, pa.RecordBatch]:
    """Fit `batch` to the relation's running schema, widening only where that is lossless.

    A DB-API driver types nothing: the relation's types come from whatever Python objects
    the first rows happened to contain. Batch 2 can therefore disagree with batch 1, and
    the two ways of handling that are both traps.

    Forcing every batch into batch 1's types — the obvious approach, and what this did
    first — **silently corrupts data**: a column whose first batch held ``1`` is typed
    `int64`, and a ``2.5`` in batch 2 is written as ``2``. No error, no warning, a wrong
    number in the output. Arrow's `safe=True` does not catch it because the value arrives
    as a Python float, never as an Arrow cast.

    The mirror failure is a column that is NULL in batch 1: the relation pins to
    `pa.null()` and the first real value dies with ``Invalid null value``. Nulls-first is
    an ordinary shape — any nullable column whose NULLs sort to the front — so that made
    perfectly good relations unreadable.

    So: promote `null` to a real type freely (nothing is lost, and the batches already
    emitted held only nulls), and refuse anything else. A genuine int-to-float widening
    cannot be applied retroactively to batches already handed downstream, and guessing is
    what produced the truncation. `schema_override=` states the types up front and skips
    this path entirely.

    Args:
        running: The relation's schema so far, or None before the first batch.
        batch: The freshly-inferred batch.

    Returns:
        The updated running schema and the batch cast to it.

    Raises:
        BackendError: If the batch's types conflict with the running schema in a way that
            cannot be resolved without either losing data or retyping what was already
            emitted.
    """
    if running is None:
        return batch.schema, batch
    if batch.schema == running:
        return running, batch

    fields: list[pa.Field] = []
    arrays: list[pa.Array] = []
    for index, field_ in enumerate(running):
        incoming = batch.column(index)
        if pa.types.is_null(field_.type) and not pa.types.is_null(incoming.type):
            # Safe promotion: every value emitted for this column so far was NULL, so
            # adopting the incoming type retypes nothing that has already been handed out.
            fields.append(batch.schema.field(index))
            arrays.append(incoming)
            continue
        try:
            # `cast` is safe by default, so it raises rather than truncating. That is the
            # whole guard: int64 under a double relation converts exactly and is allowed,
            # while a double under an int64 relation raises instead of dropping the
            # fraction. Arrow decides what is lossless; this code only decides what to do
            # about it.
            arrays.append(incoming.cast(field_.type))
        except pa.ArrowInvalid as exc:
            raise BackendError(
                f"column {field_.name!r} changed type mid-read: batches so far were "
                f"{field_.type}, this batch is {incoming.type}, and converting would lose "
                "data. A DB-API driver reports no types, so the relation's types are "
                "inferred from the first rows and cannot be widened once earlier batches "
                "have been emitted. Pass schema_override= with the intended Arrow schema, "
                "or CAST the column in your query."
            ) from exc
        fields.append(field_)
    unified = pa.schema(fields)
    return unified, pa.RecordBatch.from_arrays(arrays, schema=unified)
