"""One-vs-rest multiclass classification, and the guard that made it necessary.

`LogisticRegression` fits a single weight vector. Given a three-class target it used to fit
anyway - IRLS reads ``label - probability`` as a residual whatever the label is - and returned
a model that predicted one class for every row. Nothing raised, and the prediction column
looked exactly like a working one.

So these tests come in two halves. The first pins the guard: the shapes that must be rejected,
and just as importantly the binary shapes that must still be accepted, because a guard that
over-fires would break every two-class model in the wild. The second covers the replacement.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError
from batcher.ml import LogisticRegression, OneVsRestClassifier

pytestmark = pytest.mark.unit


def _three_classes() -> bt.Dataset:
    return bt.from_pydict(
        {
            "x": [0.0, 1.0, 2.0, 8.0, 9.0, 10.0, 20.0, 21.0, 22.0],
            "y": [0, 0, 0, 1, 1, 1, 2, 2, 2],
        }
    )


# --------------------------------------------------------------------------------------
# The guard on LogisticRegression
# --------------------------------------------------------------------------------------


def test_a_three_class_target_is_rejected_rather_than_silently_mislearned() -> None:
    with pytest.raises(PlanError, match="binary target"):
        LogisticRegression(["x"], "y").fit(_three_classes())


def test_the_rejection_names_the_offending_value_and_the_way_out() -> None:
    with pytest.raises(PlanError) as caught:
        LogisticRegression(["x"], "y").fit(_three_classes())
    message = str(caught.value)
    assert "'y'" in message, "the message must name the column"
    assert "2" in message, "the message must name a value that does not belong"
    assert "OneVsRestClassifier" in message, "the message must name the replacement"


def test_labels_that_are_neither_zero_nor_one_are_rejected_even_when_there_are_only_two():
    """Two classes of 5 and 7 are as unlearnable here as three are, and were as silent."""
    ds = bt.from_pydict({"x": [0.0, 1.0, 9.0, 10.0], "y": [5, 5, 7, 7]})
    with pytest.raises(PlanError, match="binary target"):
        LogisticRegression(["x"], "y").fit(ds)


@pytest.mark.parametrize(
    ("labels", "what"),
    [
        ([0, 0, 1, 1], "integers"),
        ([0.0, 0.0, 1.0, 1.0], "floats"),
        ([False, False, True, True], "booleans"),
        ([0, None, 1, 1], "nulls alongside a binary label"),
        ([1, 1, 1, 1], "a degenerate single class"),
    ],
)
def test_binary_targets_are_still_accepted(labels: list, what: str) -> None:
    """The guard must not fire on any spelling of 0/1, or it breaks working models."""
    ds = bt.from_pydict({"x": [-2.0, -1.0, 1.0, 2.0], "y": labels})
    model = LogisticRegression(["x"], "y").fit(ds)
    assert model.n_iter_ >= 1, f"fitting {what} should have run IRLS"


def test_the_binary_fit_is_unchanged_by_the_guard() -> None:
    ds = bt.from_pydict({"x": [-2.0, -1.0, 1.0, 2.0], "y": [0, 0, 1, 1]})
    assert LogisticRegression(["x"], "y").fit(ds).coef_[0] > 0


# --------------------------------------------------------------------------------------
# OneVsRestClassifier
# --------------------------------------------------------------------------------------


def test_three_separable_classes_are_all_predicted() -> None:
    ds = _three_classes()
    model = OneVsRestClassifier(LogisticRegression, ["x"], "y").fit(ds)
    assert model.predict(ds).to_pydict()["prediction"] == [0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_it_fits_one_sub_model_per_class() -> None:
    model = OneVsRestClassifier(LogisticRegression, ["x"], "y").fit(_three_classes())
    assert model.classes_ == [0, 1, 2]
    assert len(model.estimators_) == 3


def test_string_labels_round_trip_through_prediction() -> None:
    ds = bt.from_pydict(
        {
            "x": [0.0, 1.0, 9.0, 10.0, 20.0, 21.0],
            "y": ["low", "low", "mid", "mid", "high", "high"],
        }
    )
    model = OneVsRestClassifier(LogisticRegression, ["x"], "y").fit(ds)
    assert model.predict(ds).to_pydict()["prediction"] == [
        "low",
        "low",
        "mid",
        "mid",
        "high",
        "high",
    ]


def test_the_staged_score_columns_do_not_leak_into_the_result() -> None:
    ds = _three_classes()
    out = OneVsRestClassifier(LogisticRegression, ["x"], "y").fit(ds).predict(ds)
    assert out.columns == ["x", "y", "prediction"]


def test_the_output_column_is_configurable() -> None:
    ds = _three_classes()
    model = OneVsRestClassifier(LogisticRegression, ["x"], "y", output_column="klass").fit(ds)
    assert "klass" in model.predict(ds).columns


def test_params_reach_every_sub_model() -> None:
    model = OneVsRestClassifier(LogisticRegression, ["x"], "y", params={"max_iter": 3}).fit(
        _three_classes()
    )
    assert [sub.max_iter for sub in model.estimators_] == [3, 3, 3]


def test_the_class_order_does_not_depend_on_the_order_rows_arrive_in() -> None:
    """A label ordering read off a scan would differ between partitionings of one dataset.

    That is the property that lets a model fitted on a cluster be compared, or loaded,
    against one fitted on a laptop: the sub-models must be numbered the same way.
    """
    forward = _three_classes()
    shuffled = bt.from_pydict(
        {
            "x": [20.0, 0.0, 9.0, 21.0, 1.0, 10.0, 22.0, 2.0, 8.0],
            "y": [2, 0, 1, 2, 0, 1, 2, 0, 1],
        }
    )
    first = OneVsRestClassifier(LogisticRegression, ["x"], "y").fit(forward)
    second = OneVsRestClassifier(LogisticRegression, ["x"], "y").fit(shuffled)
    assert first.classes_ == second.classes_
    assert first.predict(forward).to_pydict() == second.predict(forward).to_pydict()


def test_a_union_of_partitions_predicts_what_one_partition_does() -> None:
    """Single-node and multi-partition input must agree, which is the mergeable contract."""
    ds = _three_classes()
    split = ds.limit(4).union(
        bt.from_pydict({"x": [9.0, 10.0, 20.0, 21.0, 22.0], "y": [1, 1, 2, 2, 2]})
    )
    whole = OneVsRestClassifier(LogisticRegression, ["x"], "y").fit(ds)
    parted = OneVsRestClassifier(LogisticRegression, ["x"], "y").fit(split)
    assert whole.classes_ == parted.classes_
    assert (
        whole.predict(ds).to_pydict()["prediction"] == parted.predict(ds).to_pydict()["prediction"]
    )


def test_it_beats_the_binary_model_it_replaces() -> None:
    """The point of the class: the shape LogisticRegression cannot express, expressed."""
    ds = _three_classes()
    predicted = (
        OneVsRestClassifier(LogisticRegression, ["x"], "y")
        .fit(ds)
        .predict(ds)
        .to_pydict()["prediction"]
    )
    assert len(set(predicted)) == 3, "all three classes must be reachable"


# --------------------------------------------------------------------------------------
# Rejections
# --------------------------------------------------------------------------------------


def test_no_features_is_rejected() -> None:
    with pytest.raises(PlanError, match="at least one feature"):
        OneVsRestClassifier(LogisticRegression, [], "y")


def test_a_base_estimator_without_predict_proba_is_rejected_at_construction() -> None:
    from batcher.ml import NearestCentroid

    with pytest.raises(PlanError, match="predict_proba"):
        OneVsRestClassifier(NearestCentroid, ["x"], "y")


def test_a_single_class_target_is_rejected() -> None:
    ds = bt.from_pydict({"x": [0.0, 1.0], "y": ["only", "only"]})
    with pytest.raises(PlanError, match="nothing to tell apart"):
        OneVsRestClassifier(LogisticRegression, ["x"], "y").fit(ds)


def test_too_many_classes_is_rejected_rather_than_fitting_one_model_each() -> None:
    ds = bt.from_pydict({"x": [float(i) for i in range(10)], "y": list(range(10))})
    with pytest.raises(PlanError, match="distinct values"):
        OneVsRestClassifier(LogisticRegression, ["x"], "y", max_classes=4).fit(ds)


def test_a_missing_column_is_named() -> None:
    with pytest.raises(ColumnNotFoundError):
        OneVsRestClassifier(LogisticRegression, ["nope"], "y").fit(_three_classes())


def test_predicting_before_fitting_is_rejected() -> None:
    model = OneVsRestClassifier(LogisticRegression, ["x"], "y")
    with pytest.raises(PlanError):
        model.predict(_three_classes())


def test_it_survives_a_save_and_load() -> None:
    from batcher.ml import load_model, save_model

    ds = _three_classes()
    model = OneVsRestClassifier(LogisticRegression, ["x"], "y").fit(ds)
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = f"{directory}/ovr.json"
        save_model(model, path)
        restored = load_model(path)
    assert restored.classes_ == model.classes_
    assert restored.estimator is LogisticRegression
    assert len(restored.estimators_) == 3
    assert restored.predict(ds).to_pydict() == model.predict(ds).to_pydict()
