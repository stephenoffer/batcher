"""Saving and restoring a fitted preprocessor — the train/serve parity contract.

A preprocessor is only useful because its state is *learned once and reused*: the scaler
that standardizes a request at serving time has to hold the training set's mean, not the
request's. That means the fitted state has to outlive the process that fitted it, and how
it is written down decides whether it can be trusted six months later.

Pickle is the usual answer and the wrong one here. A pickle is opaque (you cannot read what
the model will actually do to a column), version-fragile (it breaks when a class moves or a
slot is renamed), and unsafe to load from anywhere you do not fully control. What is written
instead is plain JSON naming the class, its hyperparameters, and its learned state — a file
a person can read, a reviewer can diff, and a serving stack in another language can consume.

Round-tripping is exact for every preprocessor in this package, and `to_dict` refuses
rather than guesses on state it cannot represent, so a silently lossy save is not possible.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError

if TYPE_CHECKING:
    from batcher.ml.preprocessors.base import Preprocessor

__all__ = ["from_dict", "load", "save", "to_dict"]

#: Bumped when the on-disk shape changes in a way an older reader cannot handle. Written
#: into every document so a mismatch is an error rather than a misread field.
SCHEMA_VERSION = 1

#: JSON has no tuple and no set. Both appear in fitted state (a category list, a bounds
#: pair), so they are tagged on the way out and restored on the way in rather than being
#: silently flattened into a list — which would round-trip to a *different* object.
_TUPLE_TAG = "__tuple__"


def _encode(value: Any) -> Any:
    """Convert fitted state into JSON-representable form, tagging what JSON would lose."""
    if isinstance(value, tuple):
        return {_TUPLE_TAG: [_encode(v) for v in value]}
    if isinstance(value, list):
        return [_encode(v) for v in value]
    if isinstance(value, dict):
        # A dict keyed by anything but a string is routine here (a category set keyed by an
        # int), and JSON would coerce those keys to strings and lose the type on reload, so
        # the mapping is written as an explicit pair list instead.
        return {"__items__": [[_encode(k), _encode(v)] for k, v in value.items()]}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise PlanError(
        f"cannot serialize fitted state of type {type(value).__name__}. Preprocessor state "
        "must be built from JSON-representable values so a saved preprocessor stays readable "
        "and portable."
    )


def _decode(value: Any) -> Any:
    """Restore what `_encode` wrote, including tuples and non-string dict keys."""
    if isinstance(value, dict):
        if _TUPLE_TAG in value:
            return tuple(_decode(v) for v in value[_TUPLE_TAG])
        if "__items__" in value:
            return {_decode(k): _decode(v) for k, v in value["__items__"]}
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v) for v in value]
    return value


def _state_names(preprocessor: Preprocessor) -> list[str]:
    """The trailing-underscore attributes holding learned state (scikit-learn's convention)."""
    names: list[str] = []
    for klass in type(preprocessor).__mro__:
        for slot in getattr(klass, "__slots__", ()):
            if slot.endswith("_") and not slot.startswith("_") and slot not in names:
                names.append(slot)
    for attr in getattr(preprocessor, "__dict__", {}):
        if attr.endswith("_") and not attr.startswith("_") and attr not in names:
            names.append(attr)
    return names


def to_dict(preprocessor: Preprocessor) -> dict[str, Any]:
    """Represent a fitted preprocessor as a plain, JSON-safe dictionary.

    The document names the class, its constructor hyperparameters, and every
    trailing-underscore attribute holding learned state — which is exactly the split
    scikit-learn draws, so the result is readable by anyone who knows that convention.

    Args:
        preprocessor: The preprocessor to represent, fitted or not.

    Returns:
        A dictionary with ``version``, ``class``, ``params``, ``state``, and ``fitted``.

    Raises:
        PlanError: If any state value is not JSON-representable.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import StandardScaler, to_dict
            >>> pre = StandardScaler("x").fit(bt.from_pydict({"x": [1.0, 3.0]}))
            >>> to_dict(pre)["class"]
            'StandardScaler'
    """
    return {
        "version": SCHEMA_VERSION,
        "class": type(preprocessor).__name__,
        "params": {k: _encode(v) for k, v in preprocessor.get_params().items()},
        "state": {n: _encode(getattr(preprocessor, n)) for n in _state_names(preprocessor)},
        "fitted": bool(preprocessor.is_fitted),
    }


def _registry() -> dict[str, type]:
    """Every preprocessor class this package exports, by name.

    Built from the package's own ``__all__`` rather than a hand-kept list, so a new
    preprocessor is loadable the moment it is exported and cannot be forgotten here.
    """
    from batcher.ml import preprocessors

    classes: dict[str, type] = {}
    for name in preprocessors.__all__:
        candidate = getattr(preprocessors, name)
        if isinstance(candidate, type):
            classes[name] = candidate
    return classes


def from_dict(document: dict[str, Any]) -> Preprocessor:
    """Rebuild a preprocessor from a `to_dict` document, learned state included.

    Args:
        document: The dictionary `to_dict` produced.

    Returns:
        The reconstructed preprocessor, fitted if the saved one was.

    Raises:
        PlanError: On an unknown schema version, an unknown class name, or params the
            class no longer accepts.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import StandardScaler, from_dict, to_dict
            >>> pre = StandardScaler("x").fit(bt.from_pydict({"x": [1.0, 3.0]}))
            >>> from_dict(to_dict(pre)).mean_
            {'x': 2.0}
    """
    version = document.get("version")
    if version != SCHEMA_VERSION:
        raise PlanError(
            f"unsupported preprocessor schema version {version!r}; this build reads "
            f"version {SCHEMA_VERSION}."
        )
    name = document.get("class")
    classes = _registry()
    if name not in classes:
        from batcher._internal.errors import suggestion

        hint = suggestion(str(name), sorted(classes))
        tail = f" {hint}" if hint else ""
        raise PlanError(f"unknown preprocessor class {name!r}.{tail}")
    params = {k: _decode(v) for k, v in document.get("params", {}).items()}
    try:
        instance = classes[name](**params)
    except TypeError as exc:
        raise PlanError(
            f"{name} no longer accepts the saved parameters ({exc}). The preprocessor was "
            "saved by a different version of Batcher; re-fit and re-save it."
        ) from exc
    for attribute, value in document.get("state", {}).items():
        setattr(instance, attribute, _decode(value))
    if document.get("fitted"):
        instance._fitted = True
    return instance


def save(preprocessor: Preprocessor, path: str) -> None:
    """Write a fitted preprocessor to `path` as JSON.

    The path may be local or a cloud URI (``s3://``, ``gs://``, ``abfs://``), because a
    fitted preprocessor belongs next to the model it feeds, not on the machine that fitted
    it.

    Args:
        preprocessor: The preprocessor to write.
        path: Where to write it; a local path or a cloud URI.

    Raises:
        PlanError: If any state value is not JSON-representable.

    Examples:
        .. doctest::

            >>> import batcher as bt, tempfile, os
            >>> from batcher.ml.preprocessors import StandardScaler, load, save
            >>> pre = StandardScaler("x").fit(bt.from_pydict({"x": [1.0, 3.0]}))
            >>> target = os.path.join(tempfile.mkdtemp(), "scaler.json")
            >>> save(pre, target)
            >>> load(target).scale_
            {'x': 1.0}
    """
    from batcher.io.filesystem import resolve_filesystem

    payload = json.dumps(to_dict(preprocessor), indent=2, sort_keys=True).encode()
    filesystem = resolve_filesystem(path)
    with filesystem.atomic_writer(path) as handle:
        handle.write(payload)


def load(path: str) -> Preprocessor:
    """Read a preprocessor written by `save`.

    Args:
        path: The local path or cloud URI to read.

    Returns:
        The reconstructed preprocessor, fitted if the saved one was.

    Raises:
        PlanError: On an unreadable document, an unknown schema version, or an unknown
            class name.

    Examples:
        .. doctest::

            >>> import batcher as bt, tempfile, os
            >>> from batcher.ml.preprocessors import MinMaxScaler, load, save
            >>> pre = MinMaxScaler("x").fit(bt.from_pydict({"x": [0.0, 10.0]}))
            >>> target = os.path.join(tempfile.mkdtemp(), "scaler.json")
            >>> save(pre, target)
            >>> load(target).is_fitted
            True
    """
    from batcher.io.filesystem import resolve_filesystem

    filesystem = resolve_filesystem(path)
    with filesystem.open(path, "rb") as handle:
        raw = handle.read()
    try:
        document = json.loads(raw)
    except ValueError as exc:
        raise PlanError(f"{path!r} is not a saved preprocessor: {exc}") from exc
    return from_dict(document)
