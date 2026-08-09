"""`blend_predictions`, `out_of_fold_features` and `StackingEnsemble`.

The load-bearing test here is `test_the_meta_model_never_sees_a_self_prediction`: stacking
is only worth anything if the meta-model trains on predictions made by models that did not
see the row, and getting that wrong produces an ensemble that scores beautifully in
development and badly in use — with nothing failing in between.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError
from batcher.ml import LinearRegression, Ridge
from batcher.ml.ensemble import (
    StackingEnsemble,
    blend_predictions,
    majority_vote,
    out_of_fold_features,
)

pytestmark = pytest.mark.unit


def _ds(n: int = 40) -> bt.Dataset:
    return bt.from_pydict({"x": [float(i) for i in range(n)], "y": [2.0 * i for i in range(n)]})


def _ridge(alpha: float = 1.0):
    return (
        lambda d: Ridge(["x"], "y", alpha=alpha).fit(d),
        lambda m, d: m.predict(d),
    )


def _ols():
    return (
        lambda d: LinearRegression(["x"], "y").fit(d),
        lambda m, d: m.predict(d),
    )


def test_blend_averages_equally_by_default() -> None:
    ds = bt.from_pydict({"a": [0.0, 1.0], "b": [1.0, 1.0]})
    assert blend_predictions(ds, ["a", "b"]).to_pydict()["prediction"] == [0.5, 1.0]


def test_blend_normalizes_the_weights() -> None:
    """[3, 1] and [0.75, 0.25] must mean the same thing, so the blend keeps its scale."""
    ds = bt.from_pydict({"a": [0.0], "b": [1.0]})
    raw = blend_predictions(ds, ["a", "b"], weights=[3, 1]).to_pydict()["prediction"]
    normalized = blend_predictions(ds, ["a", "b"], weights=[0.75, 0.25]).to_pydict()["prediction"]
    assert raw == normalized == [0.25]


def test_blend_of_one_column_is_that_column() -> None:
    ds = bt.from_pydict({"a": [1.0, 2.0, 3.0]})
    assert blend_predictions(ds, ["a"]).to_pydict()["prediction"] == [1.0, 2.0, 3.0]


def test_blend_writes_where_it_is_told() -> None:
    ds = bt.from_pydict({"a": [1.0], "b": [3.0]})
    out = blend_predictions(ds, ["a", "b"], output_column="mix")
    assert out.to_pydict()["mix"] == [2.0]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"columns": []}, "at least one prediction column"),
        ({"columns": ["a", "b"], "weights": [1.0]}, "weight"),
        ({"columns": ["a", "b"], "weights": [0.0, 0.0]}, "sum to zero"),
        ({"columns": ["a", "b"], "weights": [-1.0, 2.0]}, "non-negative"),
    ],
)
def test_blend_configuration_is_validated(kwargs: dict, message: str) -> None:
    ds = bt.from_pydict({"a": [1.0], "b": [2.0]})
    with pytest.raises(PlanError, match=message):
        blend_predictions(ds, **kwargs)


def test_blend_names_a_missing_column() -> None:
    with pytest.raises(ColumnNotFoundError):
        blend_predictions(bt.from_pydict({"a": [1.0]}), ["a", "nope"])


def test_out_of_fold_features_covers_every_row_once_per_model() -> None:
    ds = _ds()
    bases = {"ridge": _ridge(), "ols": _ols()}
    features = out_of_fold_features(ds, bases, k=4, key="x")
    assert features.count() == ds.count()
    assert {"ridge", "ols", "x", "y"} == set(features.columns)


def test_out_of_fold_columns_are_not_the_prediction_column() -> None:
    """Each base's output is renamed, so the next base's predict has a clean slate."""
    features = out_of_fold_features(_ds(), {"ridge": _ridge()}, k=4, key="x")
    assert "prediction" not in features.columns


def test_the_meta_model_never_sees_a_self_prediction() -> None:
    """A base model must be fitted without the rows it then scores, in every fold."""
    seen_in_training: list[set[float]] = []
    scored: list[set[float]] = []

    def fit(train: bt.Dataset):
        seen_in_training.append(set(train.to_pydict()["x"]))
        return LinearRegression(["x"], "y").fit(train)

    def predict(model, part: bt.Dataset) -> bt.Dataset:
        scored.append(set(part.to_pydict()["x"]))
        return model.predict(part)

    out_of_fold_features(_ds(), {"m": (fit, predict)}, k=4, key="x")
    assert len(seen_in_training) == 4
    for trained, validated in zip(seen_in_training, scored, strict=True):
        assert not (trained & validated)


def test_out_of_fold_rejects_too_few_folds() -> None:
    with pytest.raises(PlanError, match="at least 2 folds"):
        out_of_fold_features(_ds(), {"ridge": _ridge()}, k=1, key="x")


def test_out_of_fold_rejects_an_empty_base_set() -> None:
    with pytest.raises(PlanError, match="at least one base model"):
        out_of_fold_features(_ds(), {}, k=4, key="x")


def test_a_base_model_that_writes_no_prediction_column_is_named() -> None:
    bases = {"broken": (lambda d: None, lambda m, d: d)}
    with pytest.raises(PlanError, match="produced no 'prediction' column"):
        out_of_fold_features(_ds(), bases, k=4, key="x")


def test_a_base_named_like_the_prediction_column_is_rejected() -> None:
    with pytest.raises(PlanError, match="collides with the prediction column"):
        out_of_fold_features(_ds(), {"prediction": _ridge()}, k=4, key="x")


def test_a_malformed_base_pair_is_rejected() -> None:
    with pytest.raises(PlanError, match=r"\(fit, predict\) pair"):
        out_of_fold_features(_ds(), {"bad": "not a pair"}, k=4, key="x")


def _meta():
    return (
        lambda d: LinearRegression(["ridge", "ols"], "y").fit(d),
        lambda m, d: m.predict(d),
    )


def test_stacking_predicts_every_row() -> None:
    ds = _ds()
    stack = StackingEnsemble({"ridge": _ridge(), "ols": _ols()}, _meta(), k=4, key="x").fit(ds)
    out = stack.predict(ds)
    assert out.count() == ds.count()
    assert "prediction" in out.columns


def test_stacking_recovers_a_linear_signal() -> None:
    ds = _ds()
    stack = StackingEnsemble({"ridge": _ridge(), "ols": _ols()}, _meta(), k=4, key="x").fit(ds)
    predicted = stack.predict(ds).to_pydict()
    errors = [abs(p - t) for p, t in zip(predicted["prediction"], predicted["y"], strict=True)]
    assert max(errors) < 1.0


def test_the_bases_are_refitted_on_the_whole_split() -> None:
    """The fold-fitted copies exist only for the features; scoring uses a full-data fit."""
    sizes: list[int] = []

    def fit(train: bt.Dataset):
        sizes.append(train.count())
        return LinearRegression(["x"], "y").fit(train)

    meta = (lambda d: LinearRegression(["m"], "y").fit(d), lambda mo, d: mo.predict(d))
    StackingEnsemble({"m": (fit, lambda mo, d: mo.predict(d))}, meta, k=4, key="x").fit(_ds())
    assert max(sizes) == 40  # the final refit saw every row
    assert sizes.count(40) == 1  # and only the final refit did


def test_stacking_before_fit_raises() -> None:
    stack = StackingEnsemble({"ridge": _ridge(), "ols": _ols()}, _meta(), k=4, key="x")
    with pytest.raises(PlanError, match="must be fitted"):
        stack.predict(_ds())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"k": 1}, "at least 2 folds"),
        ({"meta": "nope"}, r"meta must be a \(fit, predict\) pair"),
    ],
)
def test_stacking_configuration_is_validated(kwargs: dict, message: str) -> None:
    options = {"bases": {"ridge": _ridge()}, "meta": _meta(), "k": 4, "key": "x"}
    options.update(kwargs)
    bases = options.pop("bases")
    meta = options.pop("meta")
    with pytest.raises(PlanError, match=message):
        StackingEnsemble(bases, meta, **options)


def test_stacking_is_reproducible() -> None:
    ds = _ds()

    def run() -> list[float]:
        stack = StackingEnsemble(
            {"ridge": _ridge(), "ols": _ols()}, _meta(), k=4, key="x", seed=7
        ).fit(ds)
        return stack.predict(ds).sort("x").to_pydict()["prediction"]

    assert run() == run()


def test_majority_vote_picks_the_label_most_models_chose() -> None:
    ds = bt.from_pydict({"m1": ["a", "b"], "m2": ["a", "b"], "m3": ["b", "a"]})
    assert majority_vote(ds, ["m1", "m2", "m3"]).to_pydict()["prediction"] == ["a", "b"]


def test_majority_vote_weights_a_better_model_higher() -> None:
    ds = bt.from_pydict({"strong": ["a"], "weak1": ["b"], "weak2": ["b"]})
    equal = majority_vote(ds, ["strong", "weak1", "weak2"]).to_pydict()["prediction"]
    weighted = majority_vote(ds, ["strong", "weak1", "weak2"], weights=[5.0, 1.0, 1.0]).to_pydict()[
        "prediction"
    ]
    assert equal == ["b"]
    assert weighted == ["a"]


def test_majority_vote_breaks_ties_by_label_order() -> None:
    """A tie must resolve the same way every run, not by evaluation order."""
    ds = bt.from_pydict({"m1": ["a"], "m2": ["b"]})
    assert majority_vote(ds, ["m1", "m2"], labels=["a", "b"]).to_pydict()["prediction"] == ["a"]
    assert majority_vote(ds, ["m1", "m2"], labels=["b", "a"]).to_pydict()["prediction"] == ["b"]


def test_majority_vote_accepts_explicit_labels_and_stays_lazy() -> None:
    ds = bt.from_pydict({"m1": ["a", "c"], "m2": ["a", "c"]})
    out = majority_vote(ds, ["m1", "m2"], labels=["a", "b", "c"])
    assert out.to_pydict()["prediction"] == ["a", "c"]


def test_majority_vote_works_on_integer_labels() -> None:
    ds = bt.from_pydict({"m1": [0, 2], "m2": [0, 2], "m3": [2, 0]})
    assert majority_vote(ds, ["m1", "m2", "m3"]).to_pydict()["prediction"] == [0, 2]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"columns": []}, "at least one prediction column"),
        ({"columns": ["m1", "m2"], "weights": [1.0]}, "weight"),
        ({"columns": ["m1", "m2"], "weights": [-1.0, 1.0]}, "non-negative"),
    ],
)
def test_majority_vote_configuration_is_validated(kwargs: dict, message: str) -> None:
    ds = bt.from_pydict({"m1": ["a"], "m2": ["b"]})
    with pytest.raises(PlanError, match=message):
        majority_vote(ds, **kwargs)


def test_majority_vote_names_a_missing_column() -> None:
    with pytest.raises(ColumnNotFoundError):
        majority_vote(bt.from_pydict({"m1": ["a"]}), ["m1", "nope"])
