"""The JSON document format fitted objects are written in, and how it is read back.

Both halves of `batcher.ml` that persist state — the `Preprocessor` family and the
estimators — write the same shape: a version, the class name, the constructor
hyperparameters, and the learned state. Keeping one definition of that shape here is what
stops the two from drifting into subtly different files that only one loader understands.

Why JSON rather than a pickle is the same argument in both cases. A pickle is opaque (you
cannot read what the model will do to a column), version-fragile (it breaks when a class
moves or a slot is renamed), and unsafe to load from anywhere you do not fully control.
What is written instead is a document a person can read, a reviewer can diff, and a serving
stack in another language can consume.
"""

from __future__ import annotations

import json
from typing import Any

from batcher._internal.errors import PlanError

__all__ = [
    "PREPROCESSOR_TAG",
    "SCHEMA_VERSION",
    "check_version",
    "decode_value",
    "encode_value",
    "read_document",
    "state_names",
    "write_document",
]

#: Bumped when the on-disk shape changes in a way an older reader cannot handle. Written
#: into every document so a mismatch is an error rather than a misread field.
SCHEMA_VERSION = 1

#: JSON has no tuple and no set. Both appear in fitted state (a category list, a bounds
#: pair), so they are tagged on the way out and restored on the way in rather than being
#: silently flattened into a list — which would round-trip to a *different* object.
_TUPLE_TAG = "__tuple__"

#: A NumPy array in fitted state. `QuadraticDiscriminantAnalysis` keeps a precision matrix
#: per class, and a plain list would round-trip to a *list*, so the reloaded model would
#: carry a different type from the fitted one. Tagged, so the array comes back an array.
_NDARRAY_TAG = "__ndarray__"

#: A fitted object nested inside another one's parameters — `Chain`'s steps. Written as its
#: own document under this tag so a whole fitted pipeline saves and loads as one file.
PREPROCESSOR_TAG = "__preprocessor__"


def encode_value(value: Any, *, nested: Any = None) -> Any:
    """Convert fitted state into JSON-representable form, tagging what JSON would lose.

    Args:
        value: The value to encode.
        nested: A ``(object) -> dict`` encoder for a nested fitted object, or ``None`` to
            reject one. It is injected rather than imported so this module does not have to
            know about the `Preprocessor` class it would otherwise depend on.

    Returns:
        A JSON-representable structure.

    Raises:
        PlanError: If a value has no JSON representation.
    """
    if nested is not None:
        encoded = nested(value)
        if encoded is not None:
            return {PREPROCESSOR_TAG: encoded}
    numpy_value = _as_numpy(value)
    if numpy_value is not None:
        return numpy_value
    if isinstance(value, tuple):
        return {_TUPLE_TAG: [encode_value(v, nested=nested) for v in value]}
    if isinstance(value, list):
        return [encode_value(v, nested=nested) for v in value]
    if isinstance(value, dict):
        # A dict keyed by anything but a string is routine here (a category set keyed by an
        # int), and JSON would coerce those keys to strings and lose the type on reload, so
        # the mapping is written as an explicit pair list instead.
        return {
            "__items__": [
                [encode_value(k, nested=nested), encode_value(v, nested=nested)]
                for k, v in value.items()
            ]
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise PlanError(
        f"cannot serialize fitted state of type {type(value).__name__}. State must be built "
        "from JSON-representable values so a saved object stays readable and portable."
    )


def _as_numpy(value: Any) -> Any:
    """Encode a NumPy array or scalar, or return ``None`` if `value` is neither.

    NumPy is imported lazily: this module is on the import path of every saved object, and
    most of them hold nothing but plain Python values.
    """
    if type(value).__module__.split(".")[0] != "numpy":
        return None
    import numpy as np

    if isinstance(value, np.ndarray):
        return {_NDARRAY_TAG: value.tolist()}
    # A NumPy scalar. `np.float64` happens to subclass `float` and would survive anyway,
    # but `np.int64` does not subclass `int`, so it has to be unwrapped explicitly.
    return value.item()


def decode_value(value: Any, *, nested: Any = None) -> Any:
    """Restore what `encode_value` wrote, including tuples and non-string dict keys.

    Args:
        value: The encoded structure.
        nested: A ``(dict) -> object`` decoder for a nested fitted object.

    Returns:
        The restored value.
    """
    if isinstance(value, dict):
        if PREPROCESSOR_TAG in value and nested is not None:
            return nested(value[PREPROCESSOR_TAG])
        if _NDARRAY_TAG in value:
            import numpy as np

            return np.asarray(value[_NDARRAY_TAG])
        if _TUPLE_TAG in value:
            return tuple(decode_value(v, nested=nested) for v in value[_TUPLE_TAG])
        if "__items__" in value:
            return {
                decode_value(k, nested=nested): decode_value(v, nested=nested)
                for k, v in value["__items__"]
            }
        return {k: decode_value(v, nested=nested) for k, v in value.items()}
    if isinstance(value, list):
        return [decode_value(v, nested=nested) for v in value]
    return value


def state_names(obj: object) -> list[str]:
    """The trailing-underscore attributes holding learned state (scikit-learn's convention).

    Args:
        obj: The fitted object to inspect.

    Returns:
        The attribute names carrying learned state, in declaration order.
    """
    names: list[str] = []
    for klass in type(obj).__mro__:
        for slot in getattr(klass, "__slots__", ()):
            if slot.endswith("_") and not slot.startswith("_") and slot not in names:
                names.append(slot)
    for attr in getattr(obj, "__dict__", {}):
        if attr.endswith("_") and not attr.startswith("_") and attr not in names:
            names.append(attr)
    return names


def check_version(document: dict[str, Any]) -> None:
    """Raise unless `document` was written by a schema this build can read.

    Args:
        document: The parsed document.

    Raises:
        PlanError: On an unknown schema version.
    """
    version = document.get("version")
    if version != SCHEMA_VERSION:
        raise PlanError(
            f"unsupported schema version {version!r}; this build reads version {SCHEMA_VERSION}."
        )


def write_document(document: dict[str, Any], path: str) -> None:
    """Write `document` to `path` as indented JSON, atomically.

    Args:
        document: The document to write.
        path: A local path or a cloud URI.
    """
    from batcher.io.filesystem import resolve_filesystem

    payload = json.dumps(document, indent=2, sort_keys=True).encode()
    filesystem = resolve_filesystem(path)
    with filesystem.atomic_writer(path) as handle:
        handle.write(payload)


def read_document(path: str) -> dict[str, Any]:
    """Read a document written by `write_document`.

    Args:
        path: The local path or cloud URI to read.

    Returns:
        The parsed document.

    Raises:
        PlanError: If the file is not readable JSON.
    """
    from batcher.io.filesystem import resolve_filesystem

    filesystem = resolve_filesystem(path)
    with filesystem.open(path, "rb") as handle:
        raw = handle.read()
    try:
        return json.loads(raw)
    except ValueError as exc:
        raise PlanError(f"{path!r} is not a saved Batcher object: {exc}") from exc
