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
        if (
            isinstance(candidate, type)
            and callable(getattr(candidate, "fit", None))
            and callable(getattr(candidate, "predict", None))
        ):
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
    return {
        "version": SCHEMA_VERSION,
        "class": type(model).__name__,
        "params": {k: encode_value(v) for k, v in _parameters(model).items()},
        "state": {n: encode_value(getattr(model, n)) for n in state_names(model)},
    }


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
    params = {k: decode_value(v) for k, v in document.get("params", {}).items()}
    try:
        model = classes[name](**params)
    except TypeError as exc:
        raise PlanError(
            f"{name} no longer accepts the saved parameters ({exc}). The model was saved by "
            "a different version of Batcher; re-fit and re-save it."
        ) from exc
    for attribute, value in document.get("state", {}).items():
        setattr(model, attribute, decode_value(value))
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
