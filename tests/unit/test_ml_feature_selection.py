"""The feature-selection preprocessors: what they keep, what they drop, and what leaks.

The interesting property of a selector is not that it picks the right column on a toy
frame — it is that the pick is *fitted state*. A selector that re-derives its choice on
every `transform` produces a different feature set for the training and validation splits,
which fails silently and inflates the score. Several tests below exist only to pin that.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml import Lasso, LinearRegression
from batcher.ml.preprocessors import (
    RFE,
    DropCorrelated,
    Preprocessor,
    SelectFromModel,
    SelectKBest,
    SelectPercentile,
    feature_importances,
)

pytestmark = pytest.mark.unit


def _classification() -> bt.Dataset:
    return bt.from_pydict(
        {
            "y": ["a", "a", "a", "b", "b", "b"],
            "signal": [1.0, 1.1, 0.9, 9.0, 9.2, 8.8],
            "weak": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
            "noise": [5.0, 1.0, 5.0, 1.0, 5.0, 1.0],
        }
    )


def _regression() -> bt.Dataset:
    return bt.from_pydict(
        {
            "y": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "b": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            "c": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }
    )


def test_select_k_best_keeps_the_strongest_feature() -> None:
    fitted = SelectKBest("y", k=1).fit(_classification())
    assert fitted.selected_ == ["signal"]
    assert set(fitted.scores_) == {"signal", "weak", "noise"}


def test_select_k_best_keeps_the_target_and_drops_only_rejected_features() -> None:
    out = SelectKBest("y", k=1).fit_transform(_classification())
    assert out.columns == ["y", "signal"]


def test_a_non_candidate_column_survives_selection() -> None:
    """Anything `features` did not name was never scored, so it cannot be rejected."""
    ds = _classification().with_columns(row_id=bt.col("weak"))
    out = SelectKBest("y", k=1, features=["signal", "noise"]).fit_transform(ds)
    assert set(out.columns) == {"y", "signal", "weak", "row_id"}


def test_k_above_the_feature_count_keeps_everything() -> None:
    out = SelectKBest("y", k=99).fit_transform(_classification())
    assert set(out.columns) == {"y", "signal", "weak", "noise"}


def test_select_k_best_accepts_a_regression_scorer() -> None:
    fitted = SelectKBest("y", k=1, score_func="f_regression").fit(_regression())
    assert fitted.selected_ == ["a"]


def test_select_k_best_accepts_a_custom_scorer() -> None:
    calls: list[tuple] = []

    def scorer(ds: bt.Dataset, target: str, features: list[str] | None) -> dict[str, float]:
        calls.append((target, features))
        return {"signal": 0.0, "weak": 1.0, "noise": 2.0}

    fitted = SelectKBest("y", k=1, score_func=scorer).fit(_classification())
    assert fitted.selected_ == ["noise"]
    assert calls == [("y", None)]


def test_select_percentile_sizes_the_keep_set_as_a_fraction() -> None:
    fitted = SelectPercentile("y", percentile=34).fit(_classification())
    assert fitted.selected_ == ["signal"]


def test_select_percentile_always_keeps_at_least_one_feature() -> None:
    fitted = SelectPercentile("y", percentile=1).fit(_classification())
    assert len(fitted.selected_) == 1


def test_the_selection_is_fitted_state_not_recomputed() -> None:
    """The held-out split must be pruned by the training split's choice, whatever it scores."""
    fitted = SelectKBest("y", k=1).fit(_classification())
    flipped = bt.from_pydict(
        {
            "y": ["a", "a", "b", "b"],
            "signal": [1.0, 1.0, 1.0, 1.0],
            "weak": [0.0, 0.0, 9.0, 9.0],
            "noise": [1.0, 2.0, 3.0, 4.0],
        }
    )
    assert fitted.transform(flipped).columns == ["y", "signal"]


def test_transform_before_fit_names_the_class() -> None:
    for selector in (SelectKBest("y"), SelectPercentile("y"), DropCorrelated()):
        with pytest.raises(PlanError, match="must be fitted"):
            selector.transform(_classification())


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: SelectKBest("y", k=0), "k must be at least 1"),
        (lambda: SelectPercentile("y", percentile=0), "percentile must be in"),
        (lambda: SelectPercentile("y", percentile=101), "percentile must be in"),
        (lambda: SelectKBest(["y"]), "single target column"),
        (lambda: DropCorrelated(threshold=0), "threshold must be in"),
        (lambda: SelectFromModel({}, threshold="mode"), "threshold must be"),
        (lambda: SelectFromModel({}, max_features=0), "max_features must be at least 1"),
    ],
)
def test_bad_configuration_is_rejected_with_the_reason(factory, message: str) -> None:
    with pytest.raises(PlanError, match=message):
        factory()


def test_an_unknown_scorer_lists_the_real_ones() -> None:
    with pytest.raises(PlanError, match="f_classif"):
        SelectKBest("y", score_func="anova").fit(_classification())


def test_a_missing_target_or_feature_is_named() -> None:
    with pytest.raises(PlanError, match="target 'nope'"):
        SelectKBest("nope").fit(_classification())
    with pytest.raises(PlanError, match=r"no such column\(s\) \['nope'\]"):
        SelectKBest("y", features=["signal", "nope"]).fit(_classification())


def test_drop_correlated_removes_the_duplicate_column() -> None:
    ds = bt.from_pydict(
        {"a": [1.0, 2.0, 3.0, 4.0], "a_copy": [2.0, 4.0, 6.0, 8.0], "b": [1.0, 0.0, 1.0, 0.0]}
    )
    fitted = DropCorrelated(threshold=0.95).fit(ds)
    assert fitted.dropped_ == ["a_copy"]
    assert fitted.transform(ds).columns == ["a", "b"]


def test_drop_correlated_protects_a_kept_column_by_dropping_its_partner() -> None:
    ds = bt.from_pydict({"a": [1.0, 2.0, 3.0], "a_copy": [2.0, 4.0, 6.0]})
    assert DropCorrelated(keep=["a_copy"]).fit(ds).dropped_ == ["a"]


def test_feature_importances_reads_a_batcher_estimator() -> None:
    model = LinearRegression(["a", "b", "c"], "y").fit(_regression())
    importances = feature_importances(model)
    assert set(importances) == {"a", "b", "c"}
    assert all(value >= 0 for value in importances.values())


def test_feature_importances_rejects_an_object_with_none() -> None:
    with pytest.raises(PlanError, match="exposes no feature importances"):
        feature_importances(object())


def test_select_from_model_keeps_what_an_l1_penalty_left_standing() -> None:
    ds = _regression()
    model = Lasso(["a", "b", "c"], "y", alpha=0.5).fit(ds)
    fitted = SelectFromModel(model).fit(ds)
    assert "a" in fitted.selected_
    assert set(fitted.transform(ds).columns) >= {"a", "y"}


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [(0.0, ["a", "b"]), (1.5, ["b"]), ("mean", ["b"]), ("median", ["b"])],
)
def test_select_from_model_threshold_forms(threshold, expected: list[str]) -> None:
    fitted = SelectFromModel({"a": 1.0, "b": 3.0, "c": 0.0}, threshold=threshold).fit(
        bt.from_pydict({"a": [1.0]})
    )
    assert fitted.selected_ == expected


def test_select_from_model_max_features_caps_the_keep_set() -> None:
    fitted = SelectFromModel({"a": 1.0, "b": 3.0, "c": 2.0}, max_features=2).fit(
        bt.from_pydict({"a": [1.0]})
    )
    assert fitted.selected_ == ["b", "c"]


def test_rfe_eliminates_down_to_the_requested_count() -> None:
    ds = _regression()
    rfe = RFE(
        lambda d, f: LinearRegression(list(f), "y").fit(d),
        features=["a", "b", "c"],
        n_features=1,
    )
    fitted = rfe.fit(ds)
    assert fitted.selected_ == ["a"]
    assert fitted.ranking_["a"] == 1
    assert fitted.ranking_["b"] > 1


def test_rfe_refits_once_per_elimination_round() -> None:
    seen: list[int] = []

    def fit_model(ds: bt.Dataset, features) -> dict[str, float]:
        seen.append(len(features))
        return {name: float(name == "a") for name in features}

    RFE(fit_model, features=["a", "b", "c", "d"], n_features=1).fit(_regression())
    assert seen == [4, 3, 2]


def test_rfe_step_drops_several_features_per_round() -> None:
    seen: list[int] = []

    def fit_model(ds: bt.Dataset, features) -> dict[str, float]:
        seen.append(len(features))
        return {name: float(name == "a") for name in features}

    RFE(fit_model, features=["a", "b", "c", "d", "e"], n_features=1, step=2).fit(_regression())
    assert seen == [5, 3]


def test_rfe_never_overshoots_the_requested_count() -> None:
    fit_model = lambda ds, features: {name: float(name == "a") for name in features}  # noqa: E731
    fitted = RFE(fit_model, features=["a", "b", "c"], n_features=2, step=5).fit(_regression())
    assert len(fitted.selected_) == 2


def test_rfe_rejects_a_model_that_ignores_the_feature_list() -> None:
    rfe = RFE(lambda ds, features: {"a": 1.0}, features=["a", "b"], n_features=1)
    with pytest.raises(PlanError, match="reported no importance"):
        rfe.fit(_regression())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"features": []}, "at least one column"),
        ({"features": ["a"], "n_features": 2}, "n_features must be between"),
        ({"features": ["a", "b"], "step": 0}, "step must be"),
        ({"features": ["a", "b"], "step": 1.5}, "step must be"),
    ],
)
def test_rfe_configuration_is_validated(kwargs: dict, message: str) -> None:
    with pytest.raises(PlanError, match=message):
        RFE(lambda ds, features: {}, **kwargs)


def test_rfe_rejects_a_non_callable_model() -> None:
    with pytest.raises(PlanError, match="fit_model must be"):
        RFE("not callable", features=["a"])


@pytest.mark.parametrize(
    "selector",
    [
        SelectKBest("y", k=1),
        SelectPercentile("y", percentile=50),
        DropCorrelated(threshold=0.99),
    ],
)
def test_a_fitted_selector_round_trips_through_save(selector: Preprocessor, tmp_path) -> None:
    ds = _classification()
    fitted = selector.fit(ds)
    target = str(tmp_path / "selector.json")
    fitted.save(target)
    restored = Preprocessor.load(target)
    assert restored.transform(ds).columns == fitted.transform(ds).columns


def test_selectors_compose_in_a_chain() -> None:
    from batcher.ml.preprocessors import Chain, StandardScaler

    ds = _classification()
    chained = Chain(SelectKBest("y", k=2), StandardScaler(["signal"])).fit_transform(ds)
    assert set(chained.columns) == {"y", "signal", "weak"}
