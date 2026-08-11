"""Why a Python value cannot become an Arrow column, and what to do about it.

Every column crossing the engine boundary must be an Arrow type. When one cannot be,
pyarrow raises a message that quotes the offending *value* and its class and names neither
the column it came from nor the fix — and for the Python workloads this engine targets, that
is the first error a user meets. A UUID primary key, an enum member, a PIL image, a torch
tensor: all common, all with a one-line answer, all reported today as
``did not recognize Python value type when inferring an Arrow data type``.

The alternative some engines take — silently falling back to a pickled object column — is
worse, and deliberately not taken here: the field guides put it at a 10-100x slowdown on
every downstream transfer, discovered only by eyeballing the schema. Failing loudly is the
right call. Failing loudly *and* unhelpfully is not.

This lives in `interop` because both sides of the boundary need it and neither can see the
other: `core.udf` diagnoses what a user's `map_batches` returned, and `api.session` diagnoses
what a user handed a constructor. `core` may not import the public API, so the shared answer
has to sit below both.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pyarrow as pa

__all__ = ["describe_unconvertible", "find_unconvertible_column"]

#: ``(module, class, remedy)`` for the object kinds that most often reach Arrow un-converted.
#: Matched on the *element* type of a column, by module and class name rather than by
#: importing anything, so a user with none of these installed pays nothing.
_REMEDIES: tuple[tuple[str, str, str], ...] = (
    ("torch", "Tensor", "call `.cpu().numpy()` on it — a NumPy array becomes a tensor column"),
    ("PIL", "Image", "convert it with `np.asarray(img)`, or keep the encoded bytes instead"),
    ("pandas", "DataFrame", "pass its columns individually rather than the frame object"),
    ("pandas", "Series", "call `.to_numpy()` on it"),
    (
        "uuid",
        "UUID",
        "pass `str(u)` for a text column, or `u.bytes` for a 16-byte binary one — Arrow has "
        "no inferred UUID type",
    ),
    ("pathlib", "PosixPath", "pass `str(path)`"),
    ("pathlib", "WindowsPath", "pass `str(path)`"),
)


def _element(value: object) -> object:
    """One element of a column-like `value`, or the value itself when it is not iterable."""
    try:
        return next(iter(value))  # type: ignore[call-overload]
    except Exception:
        return value


def _enum_remedy(item: object) -> str | None:
    """The fix for an `enum` member, which is matched by base class rather than by name."""
    import enum

    if isinstance(item, enum.Enum):
        kind = type(item.value).__name__
        return f"pass its `.value` (a {kind} here) — Arrow has no enum type"
    return None


def _ragged_shapes(value: object) -> str | None:
    """``"(2, 2) and (3, 3)"`` when a column holds NumPy arrays of differing shape.

    Arrow has no variable-shape tensor type, so a column of mixed-resolution arrays cannot be
    typed at all — and "convert it to an ndarray" is actively wrong advice there, because the
    caller already passed ndarrays. Naming the two shapes is the whole diagnosis: it is the
    mixed-resolution image case the multimodal guides flag.
    """
    try:
        import numpy as np

        shapes = {a.shape for a in value if isinstance(a, np.ndarray)}  # type: ignore[union-attr]
    except Exception:
        return None
    if len(shapes) < 2:
        return None
    listed = sorted(shapes, key=str)[:2]
    return " and ".join(str(s) for s in listed)


def _sample_type(value: object) -> str:
    """``"a sequence of PIL.Image"``-style description of what a column actually holds."""
    item = _element(value)
    if item is value:
        return f"a {type(value).__name__}"
    return f"a sequence of {type(item).__module__}.{type(item).__name__}"


def _remedy(value: object) -> str:
    """The one-line fix for this column's element type, or a generic one."""
    ragged = _ragged_shapes(value)
    if ragged is not None:
        return (
            f"The arrays have different shapes ({ragged}), so there is no one tensor type "
            f"that fits them — the mixed-resolution case. Resize or pad them to a common "
            f"shape, or keep the encoded bytes and decode downstream."
        )
    item = _element(value)
    enum_fix = _enum_remedy(item)
    if enum_fix is not None:
        return f"For an enum member, {enum_fix}."
    module, name = type(item).__module__.split(".")[0], type(item).__name__
    for mod, cls, fix in _REMEDIES:
        if module == mod and name == cls:
            return f"For a {mod}.{cls}, {fix}."
    return "Convert it to an Arrow-native type (a number, string, bytes, list, or ndarray)."


def describe_unconvertible(column: str, value: object) -> str:
    """A sentence naming what `column` holds and the one-line fix for it.

    Phrased as a clause a caller can prefix — ``f"{caller}(): {described}"`` on the way in,
    ``f"... a batch where {described}"`` on the way out — so both sides read as one sentence
    without either owning the wording.

    Args:
        column: The column name to quote.
        value: The values that failed to convert.

    Returns:
        A message of the form "column 'x' holds a sequence of uuid.UUID, which Arrow cannot
        represent. For a uuid.UUID, ...".
    """
    return (
        f"column {column!r} holds {_sample_type(value)}, which Arrow cannot represent. "
        f"{_remedy(value)}"
    )


def find_unconvertible_column(columns: Mapping[str, Any]) -> str | None:
    """The name of the first column in `columns` that Arrow cannot type, or `None`.

    Found by converting each column alone, which only ever runs on an error path — the caller
    already knows the batch as a whole failed and is working out which part of it did.
    """
    for name, value in columns.items():
        if isinstance(value, pa.Array | pa.ChunkedArray):
            continue
        try:
            pa.array(value)
        except Exception:
            return name
    return None
