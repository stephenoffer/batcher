"""Saving and restoring a fitted preprocessor — the train/serve parity contract.

A preprocessor is only useful because its state is *learned once and reused*: the scaler
that standardizes a request at serving time has to hold the training set's mean, not the
request's. That means the fitted state has to outlive the process that fitted it.

The document format — version, class, constructor parameters, learned state — and the
reasons for choosing JSON over a pickle live in `batcher.ml.persistence.document`, which the
estimator half uses too. What is specific to a preprocessor, and therefore lives here, is
the registry of loadable classes and the handling of a preprocessor nested inside another
one's parameters.

Round-tripping is exact for every preprocessor in this package, and `to_dict` refuses
rather than guesses on state it cannot represent, so a silently lossy save is not possible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.ml.persistence.document import (
    SCHEMA_VERSION,
    check_version,
    decode_value,
    encode_value,
    read_document,
    state_names,
    write_document,
)

if TYPE_CHECKING:
    from batcher.ml.preprocessors.base import Preprocessor

__all__ = ["from_dict", "load", "save", "to_dict"]


def _nested_encoder(value: Any) -> Any:
    """Encode a nested `Preprocessor` as its own document, or return ``None`` for anything else.

    `Chain` holds its steps as constructor parameters, so a fitted pipeline has to save as
    one file. Without this, `to_dict(Chain(...))` failed with "cannot serialize fitted state
    of type StandardScaler" — the chain named the one value the encoder had no case for, and
    the pipeline was the only preprocessor that could not be saved.
    """
    from batcher.ml.preprocessors.base import Preprocessor

    return to_dict(value) if isinstance(value, Preprocessor) else None


def _encode(value: Any) -> Any:
    """Encode one value, understanding a nested preprocessor."""
    return encode_value(value, nested=_nested_encoder)


def _decode(value: Any) -> Any:
    """Decode one value, rebuilding a nested preprocessor."""
    return decode_value(value, nested=from_dict)


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
        "state": {n: _encode(getattr(preprocessor, n)) for n in state_names(preprocessor)},
        "fitted": bool(preprocessor.is_fitted),
    }


def _registry() -> dict[str, type]:
    """Every preprocessor a user can reach, by name.

    Built from ``__all__`` rather than a hand-kept list, so a new preprocessor is loadable
    the moment it is exported and cannot be forgotten here.

    Two ``__all__``s, because one was not enough. A `Preprocessor` does not have to live in
    this package to be one: `OutlierClipper` sits in `batcher.ml.outliers` beside the outlier
    functions it belongs with, is exported from `batcher.ml` like every other preprocessor,
    and subclasses the same base — but scanning only this package's exports meant it was the
    one preprocessor `save`/`load` could not reconstruct. Fitting it into a pipeline and
    saving that pipeline succeeded; loading it back failed with "unknown preprocessor class",
    which is the worst possible time to find out. Anything reachable from `batcher.ml` that
    *is* a `Preprocessor` is therefore registered too, wherever its module happens to be.
    """
    from batcher.ml import preprocessors

    classes: dict[str, type] = {}
    for name in preprocessors.__all__:
        candidate = getattr(preprocessors, name)
        if isinstance(candidate, type):
            classes[name] = candidate

    import batcher.ml as ml_package
    from batcher.ml.preprocessors.base import Preprocessor

    for name in getattr(ml_package, "__all__", ()):
        if name in classes:
            continue
        candidate = getattr(ml_package, name, None)
        if isinstance(candidate, type) and issubclass(candidate, Preprocessor):
            classes[name] = candidate
    return classes


def _split_var_positional(klass: type, params: dict[str, Any]) -> tuple[tuple, dict[str, Any]]:
    """Pull the value for `klass`'s ``*args`` parameter out of `params`, if it has one.

    `get_params` reports every constructor parameter by name, including one declared
    ``*steps`` — and a name is exactly how a var-positional parameter cannot be passed.
    `Chain` is the case: its steps round-tripped into the document correctly and then
    `Chain(steps=[...])` raised ``TypeError``, reported as "no longer accepts the saved
    parameters", which points at a version mismatch that had not happened.

    Reading the signature rather than special-casing `Chain` keeps this true for any future
    preprocessor that takes its inputs positionally.
    """
    import inspect

    try:
        signature = inspect.signature(klass.__init__)
    except (TypeError, ValueError):  # a C-level or otherwise unreadable __init__
        return (), params
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL and parameter.name in params:
            rest = dict(params)
            value = rest.pop(parameter.name)
            return tuple(value), rest
    return (), params


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
    check_version(document)
    name = document.get("class")
    classes = _registry()
    if name not in classes:
        from batcher._internal.errors import suggestion

        hint = suggestion(str(name), sorted(classes))
        tail = f" {hint}" if hint else ""
        raise PlanError(f"unknown preprocessor class {name!r}.{tail}")
    params = {k: _decode(v) for k, v in document.get("params", {}).items()}
    positional, params = _split_var_positional(classes[name], params)
    try:
        instance = classes[name](*positional, **params)
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
    write_document(to_dict(preprocessor), path)


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
    try:
        document = read_document(path)
    except PlanError as exc:
        # The shared reader says "not a saved Batcher object"; this entry point knows which
        # kind was expected, and the narrower message is what the caller can act on.
        raise PlanError(f"{path!r} is not a saved preprocessor: {exc}") from exc
    return from_dict(document)
