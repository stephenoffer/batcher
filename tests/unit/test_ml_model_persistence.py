"""`save_model` / `load_model` across every estimator `batcher.ml` exports.

The point of this file is coverage of the *registry*, not of one happy path. A persistence
layer that works for `LinearRegression` and silently cannot rebuild `GaussianMixture` is
worse than none, because the failure lands at serving time — so every exported estimator is
fitted, saved, reloaded, and required to predict identically.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml.persistence import load_model, model_from_dict, model_to_dict, save_model
from batcher.ml.persistence.models import _registry

pytestmark = pytest.mark.unit


def _regression() -> bt.Dataset:
    return bt.from_pydict(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "z": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0],
            "y": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
        }
    )


def _classification() -> bt.Dataset:
    return bt.from_pydict(
        {
            "x": [1.0, 1.2, 0.9, 8.0, 8.2, 7.9, 1.1, 8.1],
            "z": [0.0, 0.1, 0.2, 5.0, 5.1, 4.9, 0.3, 5.2],
            "label": [0, 0, 0, 1, 1, 1, 0, 1],
        }
    )


def _counts() -> bt.Dataset:
    return bt.from_pydict(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y": [1.0, 2.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


def _positive() -> bt.Dataset:
    """A strictly positive target, so a log-family transform is defined on it."""
    return bt.from_pydict(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "z": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
            "y": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        }
    )


def _build(name: str, klass: type):
    """Fit one instance of `klass`, with the arguments that estimator actually takes."""
    from batcher.ml import (
        PCA,
        GaussianMixture,
        KMeans,
        TruncatedSVD,
    )

    if klass in (PCA, TruncatedSVD):
        return None, None  # preprocessors, covered by the preprocessor persistence tests
    if klass is KMeans:
        return klass(["x", "z"], n_clusters=2).fit(_classification()), _classification()
    if klass is GaussianMixture:
        return klass(["x", "z"], n_components=2).fit(_classification()), _classification()
    if name in ("DummyClassifier", "DummyRegressor"):
        # These take the target alone: a baseline has no features by definition.
        return klass("label").fit(_classification()), _classification()
    if name in ("LogisticRegression", "RidgeClassifier", "NearestCentroid"):
        return klass(["x", "z"], "label").fit(_classification()), _classification()
    if name in ("BernoulliNB", "GaussianNB", "MultinomialNB"):
        return klass(["x", "z"], "label").fit(_classification()), _classification()
    if name in ("LinearDiscriminantAnalysis", "QuadraticDiscriminantAnalysis"):
        return klass(["x", "z"], "label").fit(_classification()), _classification()
    if name == "TransformedTargetRegressor":
        from batcher.ml import LinearRegression, TransformedTargetRegressor

        wrapped = TransformedTargetRegressor(LinearRegression(["x", "z"], "y"), target="y")
        return wrapped.fit(_positive()), _positive()
    if name in ("PoissonRegressor", "GammaRegressor", "TweedieRegressor"):
        return klass(["x"], "y").fit(_counts()), _counts()
    return klass(["x", "z"], "y").fit(_regression()), _regression()


def _estimator_names() -> list[str]:
    from batcher.ml.preprocessors.base import Preprocessor

    return sorted(
        name for name, klass in _registry().items() if not issubclass(klass, Preprocessor)
    )


def test_the_registry_finds_the_estimators() -> None:
    names = _estimator_names()
    assert "LinearRegression" in names
    assert "KMeans" in names
    assert len(names) >= 10


#: An ensemble closes over the caller's own `fit`/`predict` functions, so it has no JSON
#: representation at all. That is a property of the object, not a gap in the writer, and it
#: is asserted explicitly below rather than skipped silently.
UNSAVEABLE = {"StackingEnsemble"}


@pytest.mark.parametrize("name", [n for n in _estimator_names() if n not in UNSAVEABLE])
def test_every_estimator_round_trips_and_predicts_identically(name: str, tmp_path) -> None:
    klass = _registry()[name]
    fitted, ds = _build(name, klass)
    if fitted is None:
        pytest.skip(f"{name} is a preprocessor, covered elsewhere")
    target = str(tmp_path / f"{name}.json")
    save_model(fitted, target)
    restored = load_model(target)
    assert type(restored) is type(fitted)
    assert restored.predict(ds).to_pydict() == fitted.predict(ds).to_pydict()


def test_a_private_parameter_is_recorded_under_its_public_name() -> None:
    """`Ridge` takes `alpha` and stores `_alpha`; an attribute scan would lose it."""
    from batcher.ml import Ridge

    fitted = Ridge(["x"], "y", alpha=0.75).fit(_regression())
    document = model_to_dict(fitted)
    assert document["params"]["alpha"] == 0.75
    assert model_from_dict(document)._alpha == 0.75


def test_the_document_records_the_learned_state_not_just_the_shape() -> None:
    from batcher.ml import LinearRegression

    document = model_to_dict(LinearRegression(["x"], "y").fit(_regression()))
    assert document["state"]["coef_"] == pytest.approx([2.0], abs=1e-6)
    assert document["class"] == "LinearRegression"
    assert document["version"] == 1


def test_a_reloaded_model_needs_no_refit() -> None:
    """The whole point: predicting must not require the training data to be present."""
    from batcher.ml import LinearRegression

    fitted = LinearRegression(["x", "z"], "y").fit(_regression())
    restored = model_from_dict(model_to_dict(fitted))
    fresh = bt.from_pydict({"x": [100.0], "z": [1.0]})
    assert restored.predict(fresh).to_pydict()["prediction"] == pytest.approx(
        fitted.predict(fresh).to_pydict()["prediction"]
    )


def test_an_unknown_class_suggests_a_real_one() -> None:
    with pytest.raises(PlanError, match="unknown estimator class"):
        model_from_dict({"version": 1, "class": "LinearRegressio", "params": {}, "state": {}})


def test_an_unknown_schema_version_is_refused() -> None:
    with pytest.raises(PlanError, match="unsupported schema version"):
        model_from_dict({"version": 99, "class": "LinearRegression", "params": {}, "state": {}})


def test_parameters_the_class_no_longer_takes_are_reported() -> None:
    with pytest.raises(PlanError, match="no longer accepts the saved parameters"):
        model_from_dict(
            {
                "version": 1,
                "class": "LinearRegression",
                "params": {"features": ["x"], "target": "y", "gone": 1},
                "state": {},
            }
        )


def test_a_file_that_is_not_a_model_says_so(tmp_path) -> None:
    path = tmp_path / "junk.json"
    path.write_text("not json at all")
    with pytest.raises(PlanError, match="is not a saved Batcher object"):
        load_model(str(path))


def test_the_saved_file_is_readable_json(tmp_path) -> None:
    """A person has to be able to read what the model will do; that is why it is not a pickle."""
    import json

    from batcher.ml import LinearRegression

    target = tmp_path / "model.json"
    save_model(LinearRegression(["x"], "y").fit(_regression()), str(target))
    document = json.loads(target.read_text())
    assert document["class"] == "LinearRegression"
    assert "coef_" in document["state"]


def test_preprocessor_persistence_still_works() -> None:
    """The document helpers are now shared, so the older caller has to keep working."""
    from batcher.ml.preprocessors import StandardScaler, from_dict, to_dict

    fitted = StandardScaler("x").fit(_regression())
    assert from_dict(to_dict(fitted)).mean_ == fitted.mean_


def test_an_ensemble_of_callables_says_why_it_cannot_be_saved() -> None:
    """The bare encoder could only name the *type*; the caller needs the field and the fix."""
    from batcher.ml import LinearRegression, StackingEnsemble

    ds = bt.from_pydict({"x": [float(i) for i in range(20)], "y": [2.0 * i for i in range(20)]})
    bases = {"m": (lambda d: LinearRegression(["x"], "y").fit(d), lambda m, d: m.predict(d))}
    meta = (lambda d: LinearRegression(["m"], "y").fit(d), lambda m, d: m.predict(d))
    stack = StackingEnsemble(bases, meta, k=4, key="x").fit(ds)
    with pytest.raises(PlanError, match=r"StackingEnsemble\.bases holds a callable"):
        model_to_dict(stack)


def test_an_unserializable_field_is_named() -> None:
    """A non-callable that JSON cannot hold should still say which attribute it was."""
    from batcher.ml import LinearRegression

    fitted = LinearRegression(["x"], "y").fit(_regression())
    fitted.coef_ = object()
    with pytest.raises(PlanError, match=r"LinearRegression\.coef_ cannot be saved"):
        model_to_dict(fitted)


def test_a_class_with_its_own_persistence_is_not_claimed_by_this_writer() -> None:
    """`Pipeline` stores steps *and* a model, so a one-estimator document cannot hold it."""
    from batcher.ml import LinearRegression, Pipeline, StandardScaler

    assert "Pipeline" not in _registry()
    pipe = Pipeline(StandardScaler(["x"]), model=LinearRegression(["x"], "y")).fit(_regression())
    with pytest.raises(PlanError, match=r"call its \.save\(path\) instead"):
        save_model(pipe, "/tmp/unused.json")


def test_a_wrapper_saves_the_estimator_it_holds(tmp_path) -> None:
    """`TransformedTargetRegressor` wraps a model; both halves belong in one document."""
    from batcher.ml import LinearRegression, TransformedTargetRegressor

    ds = _positive()
    fitted = TransformedTargetRegressor(LinearRegression(["x"], "y"), target="y").fit(ds)
    target = str(tmp_path / "wrapped.json")
    save_model(fitted, target)
    restored = load_model(target)
    assert type(restored.model).__name__ == "LinearRegression"
    assert restored.predict(ds).to_pydict() == fitted.predict(ds).to_pydict()
