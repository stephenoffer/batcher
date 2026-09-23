"""Checked column casts: reshape a file's column to a declared type, or say why it cannot.

Both schema modes cast. Strict mode casts a file's column to the *first* file's type, and
the evolution modes cast it to the reconciled type. A cast that widens (``int32`` to
``int64``, ``date`` to ``timestamp``) cannot change a value. A cast that narrows or changes
kind can, and Arrow's ``safe=True`` does not catch every case: ``5`` cast to ``bool`` is
``true``, a timestamp cast to ``date32`` drops its time of day, ``0.1`` cast to ``float32``
is ``0.10000000149011612``, and a ``timestamp[UTC]`` cast to a naive timestamp forgets which
clock it was on. Each of those returned a plausible, wrong value with no error.

So a cast that is not a widening is **verified**: the result is cast back and compared with
the input, and any value that did not survive raises `SchemaError`. The check costs a second
cast, and it runs only on a column whose type differs from the declared one and is not a
widening of it, which on a well-formed directory is never.
"""

from __future__ import annotations

import pyarrow as pa

from batcher._internal.errors import SchemaError

__all__ = ["checked_cast", "require_non_null"]


def checked_cast(
    arr: pa.Array,
    to: pa.DataType,
    *,
    verify: bool,
    keep_tz_awareness: bool,
    describe: str,
    hint: str,
) -> pa.Array:
    """`arr` cast to `to`, raising `SchemaError` instead of changing a value.

    Args:
        arr: The column as the file stored it.
        to: The declared type.
        verify: Whether to prove the cast changed no value. False only for a widening,
            which cannot.
        keep_tz_awareness: Whether a timezone-aware column may become naive or the reverse.
            Strict mode refuses it, because the declared type then describes a different
            instant for the same stored number.
        describe: The start of the error message, naming the file, column and types.
        hint: What the caller can do about it, appended to the message.

    Returns:
        The cast column.

    Raises:
        SchemaError: If the cast fails, or `verify` finds a value it changed.
    """
    import pyarrow.compute as pc

    if arr.type.equals(to):
        return arr
    reason = _tz_reason(arr.type, to) if keep_tz_awareness else None
    if reason is None:
        try:
            out = pc.cast(arr, to)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError) as exc:
            reason = f"the values do not convert ({exc})"
        else:
            if not verify or _round_trips(arr, out):
                return out
            reason = "the cast would change stored values (for example " + _example(arr, out) + ")"
    raise SchemaError(f"{describe}, and {reason}. {hint}")


def require_non_null(arr: pa.Array, field: pa.Field, *, describe: str, hint: str) -> None:
    """Raise `SchemaError` when `field` is declared non-nullable and `arr` holds a null.

    Arrow does not check this when a batch is assembled; the engine does, much later, as a
    bare `ArrowException` that names neither the file nor the declaration it broke.

    Args:
        arr: The column about to be placed under `field`.
        field: The declared field.
        describe: The start of the error message, naming the file and column.
        hint: What the caller can do about it.

    Raises:
        SchemaError: If `field` is non-nullable and `arr` has a null.
    """
    if not field.nullable and arr.null_count:
        raise SchemaError(
            f"{describe} holds {arr.null_count} null(s), but the declared schema marks it "
            f"non-nullable. {hint}"
        )


def _tz_reason(src: pa.DataType, to: pa.DataType) -> str | None:
    """Why a cast between two timestamps would change what they mean, else None."""
    if not (pa.types.is_timestamp(src) and pa.types.is_timestamp(to)):
        return None
    if (src.tz is None) == (to.tz is None):
        return None
    return (
        f"a timezone-aware timestamp and a naive one denote different instants for the same "
        f"stored value, so {src} cannot become {to} without changing its meaning"
    )


def _round_trips(arr: pa.Array, out: pa.Array) -> bool:
    """Whether casting `out` back to `arr`'s type reproduces `arr` exactly."""
    import pyarrow.compute as pc

    try:
        back = pc.cast(out, arr.type, safe=False)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError):
        return False
    if back.equals(arr):
        return True
    if not pa.types.is_floating(arr.type):
        return False
    # `Array.equals` treats NaN as unequal to itself, so a float column holding a NaN would
    # otherwise never verify. Nulls must sit in the same rows, and every other row must match.
    if not arr.is_null().equals(back.is_null()):
        return False
    same = pc.or_(pc.equal(arr, back), pc.and_(pc.is_nan(arr), pc.is_nan(back)))
    return bool(pc.all(pc.fill_null(same, True)).as_py())


def _example(arr: pa.Array, out: pa.Array) -> str:
    """The first value the cast changed, as ``before -> after``, for the error message."""
    import pyarrow.compute as pc

    try:
        back = pc.cast(out, arr.type, safe=False)
        changed = pc.fill_null(pc.not_equal(arr, back), False)
        i = pc.index(changed, True).as_py()
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError):
        i = -1
    if i < 0:
        return f"{arr.type} -> {out.type}"
    return f"{arr[i].as_py()!r} -> {out[i].as_py()!r}"
