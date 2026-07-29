"""Feature-construction preprocessors and the FeatureSpec train/serve contract.

Two things get pinned here. The preprocessors are pinned to the property that makes each
worth having — an interaction is the product a linear model can't learn, a ratio nulls a
zero denominator rather than returning infinity, a variance threshold learns on train and
applies to serve. The `FeatureSpec` is pinned to the failure it exists to catch: a serving
frame that a model would score against silently while its columns are in the wrong order or
retyped.
"""

from __future__ import annotations

import math

import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError
from batcher.ml import FeatureSpec
from batcher.ml.preprocessors import (
    Binarizer,
    Chain,
    ColumnDropper,
    ColumnSelector,
    InteractionFeatures,
    RatioFeatures,
    StandardScaler,
    VarianceThreshold,
)

pytestmark = pytest.mark.unit


# --- Binarizer -------------------------------------------------------------------------


def test_binarizer_splits_at_the_threshold() -> None:
    ds = bt.from_pydict({"x": [0.2, 0.6, 0.9]})
    assert Binarizer("x", threshold=0.5).fit_transform(ds).to_pydict()["x"] == [0, 1, 1]


def test_binarizer_is_strict_above_the_threshold() -> None:
    ds = bt.from_pydict({"x": [0.5, 0.5001]})
    assert Binarizer("x", threshold=0.5).fit_transform(ds).to_pydict()["x"] == [0, 1]


def test_binarizer_is_stateless() -> None:
    pre = Binarizer("x")
    first = pre.transform(bt.from_pydict({"x": [1.0]})).to_pydict()
    second = pre.fit_transform(bt.from_pydict({"x": [1.0]})).to_pydict()
    assert first == second


# --- ColumnSelector / ColumnDropper ----------------------------------------------------


def test_column_selector_keeps_only_the_named_columns() -> None:
    ds = bt.from_pydict({"keep": [1], "drop": [2], "also_drop": [3]})
    assert ColumnSelector(["keep"]).fit_transform(ds).columns == ["keep"]


def test_column_selector_preserves_the_requested_order() -> None:
    ds = bt.from_pydict({"a": [1], "b": [2], "c": [3]})
    assert ColumnSelector(["c", "a"]).fit_transform(ds).columns == ["c", "a"]


def test_column_dropper_removes_the_named_columns() -> None:
    ds = bt.from_pydict({"feature": [1], "user_id": [42]})
    assert ColumnDropper(["user_id"]).fit_transform(ds).columns == ["feature"]


def test_selector_and_dropper_compose_in_a_chain() -> None:
    ds = bt.from_pydict({"a": [1.0, 2.0], "b": [3.0, 4.0], "id": [9, 9]})
    out = Chain(ColumnDropper(["id"]), StandardScaler(["a", "b"])).fit_transform(ds)
    assert set(out.columns) == {"a", "b"}


# --- InteractionFeatures ---------------------------------------------------------------


def test_interaction_features_build_pairwise_products() -> None:
    ds = bt.from_pydict({"a": [2.0], "b": [3.0], "c": [4.0]})
    out = InteractionFeatures(["a", "b", "c"]).fit_transform(ds).to_pydict()
    assert (out["a_x_b"], out["a_x_c"], out["b_x_c"]) == ([6.0], [8.0], [12.0])


def test_interaction_features_keep_the_source_columns() -> None:
    ds = bt.from_pydict({"a": [2.0], "b": [3.0]})
    assert set(InteractionFeatures(["a", "b"]).fit_transform(ds).columns) == {"a", "b", "a_x_b"}


def test_interaction_features_add_no_squared_term() -> None:
    ds = bt.from_pydict({"a": [2.0], "b": [3.0]})
    columns = InteractionFeatures(["a", "b"]).fit_transform(ds).columns
    assert "a_x_a" not in columns


def test_interaction_features_need_two_columns() -> None:
    with pytest.raises(PlanError, match="at least two"):
        InteractionFeatures(["a"])


# --- RatioFeatures ---------------------------------------------------------------------


def test_ratio_features_divide_the_pair() -> None:
    ds = bt.from_pydict({"errors": [4.0], "requests": [100.0]})
    out = RatioFeatures([("errors", "requests")]).fit_transform(ds).to_pydict()
    assert out["errors_per_requests"] == [0.04]


def test_ratio_features_null_a_zero_denominator() -> None:
    # The point: null, not infinity, so one bad denominator costs its own row.
    ds = bt.from_pydict({"a": [4.0, 1.0], "b": [0.0, 2.0]})
    out = RatioFeatures([("a", "b")]).fit_transform(ds).to_pydict()
    assert out["a_per_b"] == [None, 0.5]


def test_ratio_features_reject_a_malformed_pair() -> None:
    with pytest.raises(PlanError, match="numerator, denominator"):
        RatioFeatures([("a",)])


def test_ratio_features_need_a_pair() -> None:
    with pytest.raises(PlanError, match="at least one"):
        RatioFeatures([])


# --- VarianceThreshold -----------------------------------------------------------------


def test_variance_threshold_drops_a_constant_column() -> None:
    ds = bt.from_pydict({"varies": [1.0, 2.0, 3.0], "flat": [7.0, 7.0, 7.0]})
    assert VarianceThreshold(["varies", "flat"]).fit_transform(ds).columns == ["varies"]


def test_variance_threshold_learns_on_train_and_applies_to_serve() -> None:
    # The fitted decision must travel: a column dropped on train is dropped on serve, even if
    # the serving batch happens to vary.
    train = bt.from_pydict({"a": [1.0, 2.0], "b": [5.0, 5.0]})
    pre = VarianceThreshold(["a", "b"]).fit(train)
    serve = bt.from_pydict({"a": [9.0], "b": [1.0]})
    assert pre.transform(serve).columns == ["a"]


def test_variance_threshold_honors_a_nonzero_threshold() -> None:
    ds = bt.from_pydict({"low": [1.0, 1.0, 1.1], "high": [0.0, 5.0, 10.0]})
    kept = VarianceThreshold(["low", "high"], threshold=1.0).fit_transform(ds).columns
    assert kept == ["high"]


def test_variance_threshold_rejects_a_negative_threshold() -> None:
    with pytest.raises(PlanError, match="non-negative"):
        VarianceThreshold(["a"], threshold=-1.0)


# --- FeatureSpec: capture and validate -------------------------------------------------


def test_feature_spec_captures_order_and_dtypes() -> None:
    train = bt.from_pydict({"age": [30, 40], "income": [50.0, 60.0], "label": [0, 1]})
    spec = FeatureSpec.from_dataset(train, features=["age", "income"])
    assert spec.features == ["age", "income"]
    assert spec.dtypes == {"age": "int64", "income": "double"}


def test_feature_spec_excludes_the_label_by_default() -> None:
    train = bt.from_pydict({"a": [1.0], "b": [2.0], "y": [0]})
    assert FeatureSpec.from_dataset(train, exclude=["y"]).features == ["a", "b"]


def test_validate_passes_a_matching_frame() -> None:
    spec = FeatureSpec(["a", "b"], {"a": "int64", "b": "double"})
    spec.validate(bt.from_pydict({"a": [1], "b": [2.0]}))  # no raise


def test_validate_catches_a_missing_feature() -> None:
    spec = FeatureSpec(["a", "b"], {"a": "int64", "b": "double"})
    with pytest.raises(PlanError, match="missing pinned feature"):
        spec.validate(bt.from_pydict({"a": [1]}))


def test_validate_catches_a_reordered_frame() -> None:
    # The silent killer: right columns, wrong order, a model scores against garbage.
    spec = FeatureSpec(["a", "b"], {"a": "int64", "b": "double"})
    with pytest.raises(PlanError, match="order"):
        spec.validate(bt.from_pydict({"b": [2.0], "a": [1]}))


def test_validate_catches_a_retyped_column() -> None:
    spec = FeatureSpec(["a"], {"a": "int64"})
    with pytest.raises(PlanError, match="dtype mismatch"):
        spec.validate(bt.from_pydict({"a": [1.5]}))


def test_validate_can_ignore_dtypes() -> None:
    spec = FeatureSpec(["a"], {"a": "int64"})
    spec.validate(bt.from_pydict({"a": [1.5]}), check_dtypes=False)  # no raise


def test_validate_ignores_extra_columns_but_holds_the_order() -> None:
    # Extra columns are fine as long as the feature columns are present in the pinned order.
    spec = FeatureSpec(["a", "b"], {"a": "int64", "b": "double"})
    assert spec.validate(bt.from_pydict({"a": [1], "b": [2.0], "extra": ["x"]})) is None
    # ...and the order really is pinned: the same columns in the wrong order must be rejected,
    # which is the half of this test's name that nothing was checking.
    with pytest.raises(PlanError):
        spec.validate(bt.from_pydict({"b": [2.0], "a": [1], "extra": ["x"]}))


# --- FeatureSpec: align ----------------------------------------------------------------


def test_align_reorders_and_selects() -> None:
    spec = FeatureSpec(["a", "b"], {"a": "int64", "b": "double"})
    messy = bt.from_pydict({"b": [2.0], "extra": ["x"], "a": [1]})
    assert spec.align(messy).columns == ["a", "b"]


def test_align_then_validate_is_the_serving_recipe() -> None:
    spec = FeatureSpec(["a", "b"], {"a": "int64", "b": "double"})
    messy = bt.from_pydict({"b": [2.0], "extra": ["x"], "a": [1]})
    aligned = spec.align(messy)
    assert aligned.columns == ["a", "b"]
    assert spec.validate(aligned) is None
    # The recipe is what makes serving safe, so prove it was needed: the unaligned frame
    # is exactly what validate rejects.
    with pytest.raises(PlanError):
        spec.validate(messy)


def test_align_can_cast_to_the_pinned_dtypes() -> None:
    spec = FeatureSpec(["a"], {"a": "int64"})
    aligned = spec.align(bt.from_pydict({"a": [1.0, 2.0]}), cast=True)
    assert [str(dt) for dt in aligned.dtypes] == ["int64"]


def test_align_cannot_invent_a_missing_column() -> None:
    spec = FeatureSpec(["a", "b"], {"a": "int64", "b": "double"})
    with pytest.raises(PlanError, match="lacks pinned feature"):
        spec.align(bt.from_pydict({"a": [1]}))


# --- FeatureSpec: persistence ----------------------------------------------------------


def test_feature_spec_round_trips_through_json(tmp_path) -> None:
    spec = FeatureSpec(["age", "income"], {"age": "int64", "income": "double"})
    path = str(tmp_path / "spec.json")
    spec.save(path)
    restored = FeatureSpec.load(path)
    assert restored.features == spec.features
    assert restored.dtypes == spec.dtypes


def test_feature_spec_load_rejects_a_future_version() -> None:
    with pytest.raises(PlanError, match="schema version"):
        FeatureSpec.from_dict({"version": 999, "features": ["a"], "dtypes": {"a": "int64"}})


def test_feature_spec_rejects_an_unknown_feature() -> None:
    ds = bt.from_pydict({"a": [1]})
    with pytest.raises(ColumnNotFoundError):
        FeatureSpec.from_dataset(ds, features=["nope"])


def test_feature_spec_needs_a_dtype_for_every_feature() -> None:
    with pytest.raises(PlanError, match="no dtype pinned"):
        FeatureSpec(["a", "b"], {"a": "int64"})


# --- the whole contract end to end -----------------------------------------------------


def test_a_spec_pins_a_real_training_frame_against_a_messy_serving_one() -> None:
    train = bt.from_pydict({"tenure": [1, 2, 3], "charge": [10.0, 20.0, 30.0], "churn": [0, 1, 0]})
    spec = FeatureSpec.from_dataset(train, exclude=["churn"])
    # A serving frame from a different upstream: extra column, wrong order, right data.
    serve = bt.from_pydict({"charge": [15.0], "session_id": ["x"], "tenure": [2]})
    aligned = spec.align(serve)
    assert aligned.columns == ["tenure", "charge"]
    assert math.isclose(aligned.to_pydict()["charge"][0], 15.0)
    spec.validate(aligned)
