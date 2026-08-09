"""Saving and loading a fitted estimator — the half of train/serve parity that was missing.

`Preprocessor` has had `save`/`load` since it existed, for a reason that applies just as
hard to the model: fitted state has to outlive the process that fitted it. Without this you
could train a `LinearRegression` across a cluster and then have no way to move it anywhere —
the only route to a prediction was to refit, which is not a serving story.

An estimator's state follows the same convention a preprocessor's does — constructor
hyperparameters, plus scikit-learn's trailing-underscore attributes for what `fit` learned —
so the document format is shared with `preprocessors.persistence` rather than reinvented.

One difference forces a little more care here. A preprocessor's parameters can be read off
its public attributes; an estimator's cannot, because `Ridge` takes ``alpha`` and stores it
as ``_alpha``. So the parameters are read from the **constructor signature** and matched to
``self.<name>`` or ``self._<name>``, which is both more faithful and immune to the next
estimator that keeps a parameter private.
"""

from __future__ import annotations

import inspect
from typing import Any

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

__all__ = ["load_model", "model_from_dict", "model_to_dict", "save_model"]

#: A nested estimator held inside another one's parameters — `TransformedTargetRegressor`
#: wraps the regressor it reshapes the target for. Written as its own document under this
#: tag so the wrapper saves as one file, the way `Chain` already does for preprocessors.
_MODEL_TAG = "__model__"

#: tag for a field holding an estimator *class*, recorded by name and resolved through the
#: registry on load, so a wrapper that builds its own sub-models stays saveable.
_CLASS_TAG = "__estimator_class__"


def _parameters(model: object) -> dict[str, Any]:
    """The constructor hyperparameters of `model`, read from its own signature.

    Reading the signature rather than the attributes is what makes this work for an
    estimator that keeps a parameter private: `Ridge.__init__` takes ``alpha`` and stores
    ``_alpha``, so an attribute scan would either miss it or record it under a name the
    constructor will not accept.
    """
    try:
        signature = inspect.signature(type(model).__init__)
    except (TypeError, ValueError) as exc:  # a C-level or otherwise unreadable __init__
        raise PlanError(
            f"cannot read {type(model).__name__}'s constructor, so its parameters cannot be "
            "recorded. Save the values you passed and rebuild it yourself."
        ) from exc
    out: dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if name == "self" or parameter.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        for attribute in (name, f"_{name}"):
            if hasattr(model, attribute):
                out[name] = getattr(model, attribute)
                break
        else:
            if parameter.default is inspect.Parameter.empty:
                raise PlanError(
                    f"{type(model).__name__} takes a required parameter {name!r} that it "
                    "does not keep, so a saved model could not be rebuilt. This is a bug in "
                    "the estimator, not in your call."
                )
    return out


def _registry() -> dict[str, type]:
    """Every estimator a user can reach from `batcher.ml`, by name.

    Built from ``__all__`` rather than a hand-kept list, so a new estimator is loadable the
    moment it is exported and cannot be forgotten here. Membership is by shape — a class
    with both `fit` and `predict` — which is the same `Estimator` protocol the rest of the
    package already programs against.
    """
    import batcher.ml as ml_package

    classes: dict[str, type] = {}
    for name in getattr(ml_package, "__all__", ()):
        candidate = getattr(ml_package, name, None)
        if not isinstance(candidate, type):
            continue
        if not callable(getattr(candidate, "fit", None)):
            continue
        if not callable(getattr(candidate, "predict", None)):
            continue
        # A class carrying its own `save`/`load` owns a document shape this one cannot
        # write: `Pipeline` holds preprocessors *and* a model and stores both, and a
        # `Preprocessor` has its own registry. Claiming them here would produce a file
        # neither loader could read back.
        if callable(getattr(candidate, "save", None)) and callable(
            getattr(candidate, "load", None)
        ):
            continue
        classes[name] = candidate
    return classes


def model_to_dict(model: object) -> dict[str, Any]:
    """Represent a fitted estimator as a plain, JSON-safe dictionary.

    Args:
        model: The fitted estimator to represent.

    Returns:
        A dictionary with ``version``, ``class``, ``params``, and ``state``.

    Raises:
        PlanError: If a parameter or state value has no JSON representation.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import LinearRegression
            >>> from batcher.ml.persistence import model_to_dict
            >>> ds = bt.from_pydict({"x": [1.0, 2.0], "y": [2.0, 4.0]})
            >>> model_to_dict(LinearRegression(["x"], "y").fit(ds))["class"]
            'LinearRegression'
    """
    name = type(model).__name__
    return {
        "version": SCHEMA_VERSION,
        "class": name,
        "params": {k: _encode_field(v, name, k) for k, v in _parameters(model).items()},
        "state": {n: _encode_field(getattr(model, n), name, n) for n in state_names(model)},
    }


def _encode_field(value: Any, model: str, field: str) -> Any:
    """Encode one field, saying which one failed and what to do instead.

    The bare encoder can only report the offending *type*, and "cannot serialize fitted
    state of type function" tells a caller nothing about which attribute or which way out.
    A callable is the case worth naming: an ensemble that closes over user `fit`/`predict`
    pairs cannot be written as JSON at all, and no amount of retrying will change that.
    """
    if _is_estimator(value):
        # A wrapper holding another estimator saves as one document rather than obliging
        # the caller to persist the two halves and remember how they were connected.
        return {_MODEL_TAG: model_to_dict(value)}
    if isinstance(value, type):
        # A wrapper parameterized by an estimator *class* rather than an instance, the way
        # `OneVsRestClassifier` is: it must build one sub-model per class, so it cannot be
        # handed a single pre-built one. The name is recorded and resolved back through the
        # same registry, which is why an unregistered class has to fail here - writing a name
        # nothing can look up would produce a file that saves and never loads.
        if _registry().get(value.__name__) is not value:
            raise PlanError(
                f"{model}.{field} holds the class {value.__name__!r}, which is not exported "
                "from batcher.ml, so a saved model could not name it well enough to rebuild. "
                "Parameterize the wrapper with an exported estimator."
            )
        return {_CLASS_TAG: value.__name__}
    if isinstance(value, list) and any(_is_estimator(item) for item in value):
        # One-vs-rest keeps a list of fitted sub-models. Encoding them individually keeps
        # the whole ensemble in one document, matching the single-estimator case above.
        return [_encode_field(item, model, field) for item in value]
    try:
        return encode_value(value)
    except PlanError as exc:
        if callable(value) or _holds_callable(value):
            raise PlanError(
                f"{model}.{field} holds a callable, so this model cannot be saved as JSON. "
                "An ensemble that closes over your own fit/predict functions has no portable "
                "representation — save each base model with save_model() and rebuild the "
                "ensemble in code."
            ) from exc
        raise PlanError(f"{model}.{field} cannot be saved: {exc}") from exc


def _is_estimator(value: Any) -> bool:
    """Whether `value` is a fitted estimator this module knows how to write."""
    return type(value).__name__ in _registry() and callable(getattr(value, "predict", None))


def _decode_field(value: Any) -> Any:
    """Decode one field, rebuilding a nested estimator."""
    if isinstance(value, dict) and _MODEL_TAG in value:
        return model_from_dict(value[_MODEL_TAG])
    if isinstance(value, dict) and _CLASS_TAG in value:
        name = value[_CLASS_TAG]
        klass = _registry().get(name)
        if klass is None:
            raise PlanError(
                f"this model was parameterized by {name!r}, which batcher.ml no longer "
                "exports. It was saved by a version that had it; rebuild the model with an "
                "estimator this version provides."
            )
        return klass
    if isinstance(value, list) and any(
        isinstance(item, dict) and _MODEL_TAG in item for item in value
    ):
        return [_decode_field(item) for item in value]
    return decode_value(value)


def _holds_callable(value: Any) -> bool:
    """Whether `value` is a container with a callable somewhere inside it."""
    if isinstance(value, dict):
        return any(_holds_callable(v) or callable(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_holds_callable(v) or callable(v) for v in value)
    return False


def model_from_dict(document: dict[str, Any]) -> Any:
    """Rebuild an estimator from a `model_to_dict` document, learned state included.

    Args:
        document: The dictionary `model_to_dict` produced.

    Returns:
        The reconstructed estimator, ready to `predict`.

    Raises:
        PlanError: On an unknown schema version, an unknown class name, or parameters the
            class no longer accepts.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import LinearRegression
            >>> from batcher.ml.persistence import model_from_dict, model_to_dict
            >>> ds = bt.from_pydict({"x": [1.0, 2.0], "y": [2.0, 4.0]})
            >>> fitted = LinearRegression(["x"], "y").fit(ds)
            >>> model_from_dict(model_to_dict(fitted)).coef_
            [2.0]
    """
    check_version(document)
    name = document.get("class")
    classes = _registry()
    if name not in classes:
        from batcher._internal.errors import suggestion

        hint = suggestion(str(name), sorted(classes))
        tail = f" {hint}" if hint else ""
        raise PlanError(f"unknown estimator class {name!r}.{tail}")
    params = {k: _decode_field(v) for k, v in document.get("params", {}).items()}
    try:
        model = classes[name](**params)
    except TypeError as exc:
        raise PlanError(
            f"{name} no longer accepts the saved parameters ({exc}). The model was saved by "
            "a different version of Batcher; re-fit and re-save it."
        ) from exc
    for attribute, value in document.get("state", {}).items():
        setattr(model, attribute, _decode_field(value))
    return model


def save_model(model: object, path: str) -> None:
    """Write a fitted estimator to `path` as readable JSON.

    The path may be local or a cloud URI, because a fitted model belongs next to the data it
    scores rather than on the machine that fitted it.

    Args:
        model: The fitted estimator to write.
        path: Where to write it; a local path or a cloud URI.

    Raises:
        PlanError: If a parameter or state value has no JSON representation.

    Examples:
        .. doctest::

            >>> import batcher as bt, os, tempfile
            >>> from batcher.ml import LinearRegression
            >>> from batcher.ml.persistence import load_model, save_model
            >>> ds = bt.from_pydict({"x": [1.0, 2.0], "y": [2.0, 4.0]})
            >>> target = os.path.join(tempfile.mkdtemp(), "model.json")
            >>> save_model(LinearRegression(["x"], "y").fit(ds), target)
            >>> load_model(target).coef_
            [2.0]
    """
    if callable(getattr(model, "save", None)) and callable(getattr(type(model), "load", None)):
        raise PlanError(
            f"{type(model).__name__} writes its own document — call its .save(path) instead. "
            "This writer records one estimator, and that class holds more than one thing."
        )
    write_document(model_to_dict(model), path)


def load_model(path: str) -> Any:
    """Read an estimator written by `save_model`, learned state included.

    Args:
        path: The local path or cloud URI to read.

    Returns:
        The reconstructed estimator, ready to `predict`.

    Raises:
        PlanError: On an unreadable document, an unknown schema version, or an unknown
            class name.

    Examples:
        .. doctest::

            >>> import batcher as bt, os, tempfile
            >>> from batcher.ml import Ridge
            >>> from batcher.ml.persistence import load_model, save_model
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
            >>> target = os.path.join(tempfile.mkdtemp(), "ridge.json")
            >>> save_model(Ridge(["x"], "y", alpha=0.5).fit(ds), target)
            >>> type(load_model(target)).__name__
            'Ridge'
    """
    return model_from_dict(read_document(path))
