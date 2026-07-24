"""Ranking metrics, model-free feature selection, and timestamp features.

The ranking metrics are checked against reference implementations written the obvious way
in Python over the same data, because there is no ranking oracle in the dev dependencies
and "trust the SQL" is not a check. The reference is deliberately naive: sort each user's
rows, walk them, average over users.

The property that matters most and is easiest to get wrong is *averaging over queries*
rather than pooling their rows. Pooling silently rewards a model that ranks one heavy user
well and everyone else badly, and the two agree on uniform data — so the tests use groups
of different sizes, where they do not.
"""

from __future__ import annotations

import collections
import datetime as dt
import math

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError
from batcher.ml.metrics import (
    best_cost_threshold,
    best_threshold,
    classification_report,
    compare_models,
    expected_cost_curve,
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from batcher.ml.metrics.evaluate import multiclass_averages
from batcher.ml.preprocessors import (
    CyclicalEncoder,
    DateTimeFeaturizer,
    LagFeaturizer,
    RollingFeaturizer,
)
from batcher.ml.selection import (
    constant_columns,
    correlated_columns,
    feature_profile,
    feature_report,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def rankings() -> tuple[list[tuple[str, float, int]], bt.Dataset]:
    """Fifty users with ten scored candidates each, about 30% of them relevant."""
    rng = np.random.default_rng(0)
    rows = [
        (f"u{user}", float(rng.random()), int(rng.random() < 0.3))
        for user in range(50)
        for _ in range(10)
    ]
    ds = bt.from_pydict(
        {
            "user": [r[0] for r in rows],
            "s": [r[1] for r in rows],
            "rel": [r[2] for r in rows],
        }
    )
    return rows, ds


def _by_user(rows: list[tuple[str, float, int]]) -> dict[str, list[tuple[float, int]]]:
    """Group the raw rows by user, for the reference implementations."""
    grouped: dict[str, list[tuple[float, int]]] = collections.defaultdict(list)
    for user, score, relevant in rows:
        grouped[user].append((score, relevant))
    return grouped


# --- ranking metrics ------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 3, 5, 10])
def test_precision_at_k_matches_the_reference(rankings, k) -> None:
    rows, ds = rankings
    expected = np.mean(
        [sum(r for _, r in sorted(v, key=lambda x: -x[0])[:k]) / k for v in _by_user(rows).values()]
    )
    assert precision_at_k(ds, "user", "s", "rel", k=k) == pytest.approx(expected)


@pytest.mark.parametrize("k", [1, 3, 5])
def test_recall_at_k_matches_the_reference(rankings, k) -> None:
    rows, ds = rankings
    values = []
    for candidates in _by_user(rows).values():
        total = sum(r for _, r in candidates)
        if total:
            hits = sum(r for _, r in sorted(candidates, key=lambda x: -x[0])[:k])
            values.append(hits / total)
    assert recall_at_k(ds, "user", "s", "rel", k=k) == pytest.approx(np.mean(values))


def test_mean_reciprocal_rank_matches_the_reference(rankings) -> None:
    rows, ds = rankings
    total = 0.0
    for candidates in _by_user(rows).values():
        for position, (_, relevant) in enumerate(sorted(candidates, key=lambda x: -x[0]), 1):
            if relevant:
                total += 1.0 / position
                break
    expected = total / len(_by_user(rows))
    assert mean_reciprocal_rank(ds, "user", "s", "rel") == pytest.approx(expected)


@pytest.mark.parametrize("k", [3, 5, 10])
def test_ndcg_at_k_matches_the_reference(rankings, k) -> None:
    rows, ds = rankings
    values = []
    for candidates in _by_user(rows).values():
        ordered = sorted(candidates, key=lambda x: -x[0])[:k]
        gain = sum(r / math.log2(i + 1) for i, (_, r) in enumerate(ordered, 1))
        relevant = sum(r for _, r in candidates)
        ideal = sum(1 / math.log2(i + 1) for i in range(1, min(k, relevant) + 1))
        if ideal:
            values.append(gain / ideal)
    assert ndcg_at_k(ds, "user", "s", "rel", k=k) == pytest.approx(np.mean(values))


def test_a_perfect_ranking_scores_one_on_ndcg() -> None:
    ds = bt.from_pydict({"u": ["a"] * 4, "s": [0.9, 0.8, 0.2, 0.1], "rel": [1, 1, 0, 0]})
    assert ndcg_at_k(ds, "u", "s", "rel", k=4) == pytest.approx(1.0)


def test_hit_rate_counts_a_query_once_however_many_hits_it_has() -> None:
    ds = bt.from_pydict(
        {
            "u": ["a", "a", "b", "b"],
            "s": [0.9, 0.8, 0.9, 0.8],
            "rel": [1, 1, 0, 0],
        }
    )
    assert hit_rate_at_k(ds, "u", "s", "rel", k=2) == pytest.approx(0.5)


def test_a_query_with_no_relevant_item_scores_zero_on_mrr() -> None:
    # It must still count in the denominator, or MRR reports the score of the queries that
    # happened to work.
    ds = bt.from_pydict({"u": ["a", "b"], "s": [0.9, 0.9], "rel": [1, 0]})
    assert mean_reciprocal_rank(ds, "u", "s", "rel") == pytest.approx(0.5)


def test_metrics_average_over_queries_not_over_rows() -> None:
    # One user with many bad rows and one with a single good row. Pooling gives 0.1;
    # averaging over users gives 0.5, which is the honest number.
    users = ["heavy"] * 10 + ["light"]
    scores = [0.9 - 0.01 * i for i in range(10)] + [0.5]
    relevance = [0] * 10 + [1]
    ds = bt.from_pydict({"u": users, "s": scores, "rel": relevance})
    assert hit_rate_at_k(ds, "u", "s", "rel", k=1) == pytest.approx(0.5)


def test_a_ranking_metric_rejects_a_zero_cutoff(rankings) -> None:
    _, ds = rankings
    with pytest.raises(PlanError, match="k must be at least 1"):
        precision_at_k(ds, "user", "s", "rel", k=0)


def test_a_ranking_metric_names_a_missing_column(rankings) -> None:
    _, ds = rankings
    with pytest.raises(ColumnNotFoundError):
        precision_at_k(ds, "user", "nope", "rel")


# --- feature selection -----------------------------------------------------------------


def test_constant_columns_finds_a_dead_column() -> None:
    ds = bt.from_pydict({"useful": [1.0, 2.0, 3.0], "dead": [7.0, 7.0, 7.0]})
    assert constant_columns(ds) == ["dead"]


def test_constant_columns_finds_a_near_constant_flag() -> None:
    # Non-zero variance, and still a flag rather than a measurement.
    ds = bt.from_pydict({"x": [0.0] * 99 + [1.0]})
    assert constant_columns(ds, max_mode_share=0.95) == ["x"]
    assert constant_columns(ds) == []


def test_constant_columns_rejects_an_impossible_share() -> None:
    ds = bt.from_pydict({"x": [1.0]})
    with pytest.raises(PlanError, match="max_mode_share"):
        constant_columns(ds, max_mode_share=0.0)


def test_correlated_columns_drops_the_later_of_a_redundant_pair() -> None:
    ds = bt.from_pydict({"a": [1.0, 2.0, 3.0], "copy": [2.0, 4.0, 6.0], "other": [5.0, 1.0, 4.0]})
    assert correlated_columns(ds) == ["copy"]


def test_correlated_columns_is_deterministic() -> None:
    # A screen that depends on iteration order gives a different feature set every run.
    ds = bt.from_pydict(
        {"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 8.0], "c": [3.0, 6.0, 9.0, 12.0]}
    )
    assert correlated_columns(ds) == correlated_columns(ds) == ["b", "c"]


def test_correlated_columns_keeps_an_independent_pair() -> None:
    rng = np.random.default_rng(3)
    ds = bt.from_pydict({"a": rng.normal(size=200).tolist(), "b": rng.normal(size=200).tolist()})
    assert correlated_columns(ds) == []


def test_correlated_columns_needs_two_columns() -> None:
    assert correlated_columns(bt.from_pydict({"a": [1.0, 2.0]})) == []


def test_feature_report_ranks_the_predictive_feature_first() -> None:
    rng = np.random.default_rng(5)
    signal = rng.random(600)
    ds = bt.from_pydict(
        {
            "good": signal.tolist(),
            "noise": rng.random(600).tolist(),
            "y": (rng.random(600) < signal).astype(int).tolist(),
        }
    )
    report = feature_report(ds, "y", buckets=5).to_pydict()
    assert report["feature"][0] == "good"
    assert report["information_value"][0] > report["information_value"][1]


def test_feature_report_reports_the_null_rate() -> None:
    ds = bt.from_pydict({"x": [1.0, None, 3.0, None], "y": [0, 1, 0, 1]})
    report = feature_report(ds, "y", buckets=2).to_pydict()
    assert report["null_rate"][0] == pytest.approx(0.5)


def test_feature_report_survives_an_unbinnable_column() -> None:
    # A constant feature has no quantile cut points. That is a fact about the feature, not
    # a reason to fail the whole report.
    ds = bt.from_pydict({"flat": [1.0] * 8, "y": [0, 1] * 4})
    report = feature_report(ds, "y", buckets=2).to_pydict()
    assert math.isnan(report["information_value"][0])


def test_feature_report_names_a_missing_target() -> None:
    ds = bt.from_pydict({"x": [1.0]})
    with pytest.raises(ColumnNotFoundError):
        feature_report(ds, "nope")


# --- timestamp features ------------------------------------------------------------------


def test_datetime_featurizer_extracts_the_requested_parts() -> None:
    ds = bt.from_pydict({"t": [dt.datetime(2024, 3, 16, 14, 30)]})
    out = DateTimeFeaturizer("t", parts=["year", "month", "day", "hour"]).fit_transform(ds)
    got = out.to_pydict()
    assert (got["t_year"], got["t_month"], got["t_day"], got["t_hour"]) == ([2024], [3], [16], [14])


def test_datetime_featurizer_flags_a_weekend() -> None:
    ds = bt.from_pydict({"t": [dt.datetime(2024, 3, 16), dt.datetime(2024, 3, 18)]})
    out = DateTimeFeaturizer("t", parts=["is_weekend"]).fit_transform(ds)
    assert out.to_pydict()["t_is_weekend"] == [True, False]


def test_datetime_featurizer_keeps_the_original_by_default() -> None:
    ds = bt.from_pydict({"t": [dt.datetime(2024, 1, 1)]})
    assert "t" in DateTimeFeaturizer("t", parts=["year"]).fit_transform(ds).columns


def test_datetime_featurizer_can_drop_the_original() -> None:
    ds = bt.from_pydict({"t": [dt.datetime(2024, 1, 1)]})
    out = DateTimeFeaturizer("t", parts=["year"], drop_original=True).fit_transform(ds)
    assert out.columns == ["t_year"]


def test_datetime_featurizer_rejects_an_unknown_part() -> None:
    with pytest.raises(PlanError, match="unknown calendar part"):
        DateTimeFeaturizer("t", parts=["fortnight"])


def test_cyclical_encoding_puts_midnight_next_to_hour_23() -> None:
    # The whole point: an integer encoding puts them 23 apart when they are 1 apart.
    ds = bt.from_pydict(
        {
            "t": [
                dt.datetime(2024, 1, 1, 23),
                dt.datetime(2024, 1, 2, 0),
                dt.datetime(2024, 1, 2, 12),
            ]
        }
    )
    got = CyclicalEncoder("t", parts=["hour"]).fit_transform(ds).to_pydict()
    points = list(zip(got["t_hour_sin"], got["t_hour_cos"], strict=True))

    def distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    assert distance(points[0], points[1]) < distance(points[0], points[2])


def test_cyclical_encoding_lands_on_the_unit_circle() -> None:
    ds = bt.from_pydict({"t": [dt.datetime(2024, 1, 1, h) for h in range(24)]})
    got = CyclicalEncoder("t", parts=["hour"]).fit_transform(ds).to_pydict()
    for sin, cos in zip(got["t_hour_sin"], got["t_hour_cos"], strict=True):
        assert sin * sin + cos * cos == pytest.approx(1.0)


def test_cyclical_encoding_rejects_a_non_periodic_part() -> None:
    # A year does not wrap, so encoding it on a circle would be nonsense.
    with pytest.raises(PlanError, match="unknown calendar part"):
        CyclicalEncoder("t", parts=["year"])


def test_both_timestamp_preprocessors_are_stateless() -> None:
    ds = bt.from_pydict({"t": [dt.datetime(2024, 6, 1, 8)]})
    for preprocessor in (
        DateTimeFeaturizer("t", parts=["hour"]),
        CyclicalEncoder("t", parts=["hour"]),
    ):
        first = preprocessor.transform(ds).to_pydict()
        second = preprocessor.fit_transform(ds).to_pydict()
        assert first == second


# --- multi-class reporting ---------------------------------------------------------------

skm = pytest.importorskip("sklearn.metrics", reason="scikit-learn is the metric oracle")

MULTI_TRUE = ["a", "a", "b", "c", "c", "c"]
MULTI_PRED = ["a", "b", "b", "c", "c", "a"]


def _multi() -> bt.Dataset:
    return bt.from_pydict({"y": MULTI_TRUE, "p": MULTI_PRED})


def test_classification_report_matches_sklearn_per_class() -> None:
    report = classification_report(_multi(), "y", "p").sort("class").to_pydict()
    expected = skm.classification_report(MULTI_TRUE, MULTI_PRED, output_dict=True, zero_division=0)
    for label, precision, recall, f1, support in zip(
        report["class"],
        report["precision"],
        report["recall"],
        report["f1"],
        report["support"],
        strict=True,
    ):
        assert precision == pytest.approx(expected[label]["precision"])
        assert recall == pytest.approx(expected[label]["recall"])
        assert f1 == pytest.approx(expected[label]["f1-score"])
        assert support == expected[label]["support"]


def test_classification_report_includes_a_class_the_model_never_predicts() -> None:
    # The failure a single accuracy hides: a class the model ignores entirely.
    ds = bt.from_pydict({"y": ["a", "a", "rare"], "p": ["a", "a", "a"]})
    report = classification_report(ds, "y", "p").sort("class").to_pydict()
    assert "rare" in report["class"]
    assert report["recall"][report["class"].index("rare")] == 0.0


def test_classification_report_orders_by_support() -> None:
    report = classification_report(_multi(), "y", "p").to_pydict()
    assert report["support"] == sorted(report["support"], reverse=True)


def test_classification_report_refuses_an_identifier_column() -> None:
    ds = bt.from_pydict({"y": [f"id{i}" for i in range(50)], "p": [f"id{i}" for i in range(50)]})
    with pytest.raises(PlanError, match="distinct labels"):
        classification_report(ds, "y", "p", max_classes=10)


@pytest.mark.parametrize(
    ("name", "average", "metric"),
    [
        ("macro_precision", "macro", "precision"),
        ("macro_recall", "macro", "recall"),
        ("macro_f1", "macro", "f1"),
        ("weighted_precision", "weighted", "precision"),
        ("weighted_recall", "weighted", "recall"),
        ("weighted_f1", "weighted", "f1"),
    ],
)
def test_multiclass_average_matches_sklearn(name, average, metric) -> None:
    got = multiclass_averages(_multi(), "y", "p", max_classes=10)[name]
    oracle = {
        "precision": skm.precision_score,
        "recall": skm.recall_score,
        "f1": skm.f1_score,
    }[metric]
    expected = oracle(MULTI_TRUE, MULTI_PRED, average=average, zero_division=0)
    assert got == pytest.approx(expected)


def test_evaluate_multiclass_reports_the_averages() -> None:
    report = _multi().ml.evaluate("y", y_pred="p", task="multiclass")
    assert report["accuracy"] == pytest.approx(skm.accuracy_score(MULTI_TRUE, MULTI_PRED))
    assert report["macro_f1"] == pytest.approx(
        skm.f1_score(MULTI_TRUE, MULTI_PRED, average="macro", zero_division=0)
    )


def test_a_multiclass_average_cannot_be_grouped() -> None:
    ds = bt.from_pydict({"g": ["x", "x"], "y": ["a", "b"], "p": ["a", "b"]})
    with pytest.raises(PlanError, match="by="):
        ds.ml.evaluate("y", y_pred="p", task="multiclass", by="g", metrics=["macro_f1"])


# --- lag and rolling features --------------------------------------------------------------


def test_lag_reaches_the_previous_row() -> None:
    ds = bt.from_pydict({"t": [1, 2, 3], "v": [10.0, 20.0, 30.0]})
    got = LagFeaturizer("v", order_by="t", lags=[1]).fit_transform(ds).sort("t").to_pydict()
    assert got["v_lag_1"] == [None, 10.0, 20.0]


def test_lag_never_crosses_a_series_boundary() -> None:
    # Without partition_by the last row of series "a" would feed the first row of "b".
    ds = bt.from_pydict(
        {"k": ["a", "a", "b", "b"], "t": [1, 2, 1, 2], "v": [1.0, 2.0, 100.0, 200.0]}
    )
    pre = LagFeaturizer("v", order_by="t", lags=[1], partition_by="k")
    got = pre.fit_transform(ds).sort("k", "t").to_pydict()
    assert got["v_lag_1"] == [None, 1.0, None, 100.0]


def test_several_lags_build_several_columns() -> None:
    ds = bt.from_pydict({"t": list(range(10)), "v": [float(i) for i in range(10)]})
    out = LagFeaturizer("v", order_by="t", lags=[1, 7]).fit_transform(ds)
    assert {"v_lag_1", "v_lag_7"} <= set(out.columns)


def test_lag_rejects_a_zero_step() -> None:
    with pytest.raises(PlanError, match="positive row counts"):
        LagFeaturizer("v", order_by="t", lags=[0])


def test_a_rolling_window_excludes_the_current_row() -> None:
    # The leak this module exists to prevent: a window including the current row puts the
    # target's own value inside its own feature.
    ds = bt.from_pydict({"t": [1, 2, 3], "v": [10.0, 20.0, 60.0]})
    pre = RollingFeaturizer("v", order_by="t", window=2, aggregates=["mean"])
    got = pre.fit_transform(ds).sort("t").to_pydict()
    assert got["v_rolling_mean_2"] == [None, 10.0, 15.0]


def test_a_rolling_window_stays_inside_its_series() -> None:
    ds = bt.from_pydict(
        {"k": ["a", "a", "b", "b"], "t": [1, 2, 1, 2], "v": [1.0, 2.0, 100.0, 200.0]}
    )
    pre = RollingFeaturizer("v", order_by="t", window=3, partition_by="k")
    got = pre.fit_transform(ds).sort("k", "t").to_pydict()
    assert got["v_rolling_mean_3"] == [None, 1.0, None, 100.0]


def test_several_rolling_aggregates_build_several_columns() -> None:
    ds = bt.from_pydict({"t": [1, 2, 3], "v": [1.0, 2.0, 3.0]})
    out = RollingFeaturizer(
        "v", order_by="t", window=2, aggregates=["mean", "max", "count"]
    ).fit_transform(ds)
    assert {"v_rolling_mean_2", "v_rolling_max_2", "v_rolling_count_2"} <= set(out.columns)


def test_rolling_rejects_an_unknown_aggregate() -> None:
    with pytest.raises(PlanError, match="unknown rolling aggregate"):
        RollingFeaturizer("v", order_by="t", aggregates=["kurtosis"])


def test_rolling_rejects_a_zero_window() -> None:
    with pytest.raises(PlanError, match="window must be at least"):
        RollingFeaturizer("v", order_by="t", window=0)


# --- the drift accessor ---------------------------------------------------------------------


def test_the_drift_accessor_flags_a_shifted_column() -> None:
    train = bt.from_pydict({"x": [float(i) for i in range(200)]})
    today = bt.from_pydict({"x": [float(i) + 120 for i in range(200)]})
    report = today.ml.drift(train, ["x"], buckets=5).to_pydict()
    assert report["psi"][0] > 0.25
    assert report["mean_shift"][0] == pytest.approx(120.0)


def test_the_drift_accessor_defaults_to_every_numeric_column() -> None:
    train = bt.from_pydict({"x": [float(i) for i in range(50)], "label": ["a"] * 50})
    today = bt.from_pydict({"x": [float(i) for i in range(50)], "label": ["a"] * 50})
    assert today.ml.drift(train, buckets=4).to_pydict()["column"] == ["x"]


# --- choosing an operating point ------------------------------------------------------------


@pytest.fixture(scope="module")
def scored() -> tuple[np.ndarray, np.ndarray, bt.Dataset]:
    """A binary problem whose F1-optimal cutoff is nowhere near 0.5."""
    rng = np.random.default_rng(4)
    score = rng.random(2000)
    labels = (rng.random(2000) < 0.15 + 0.5 * score).astype(int)
    return labels, score, bt.from_pydict({"y": labels.tolist(), "s": score.tolist()})


def test_best_threshold_matches_a_brute_force_search(scored) -> None:
    labels, score, ds = scored
    got = best_threshold(ds, "y", "s", thresholds=100)
    candidates = [i / 100 for i in range(100)]
    expected = max(
        candidates,
        key=lambda t: skm.f1_score(labels, (score >= t).astype(int), zero_division=0),
    )
    assert got["threshold"] == pytest.approx(expected)
    assert got["f1"] == pytest.approx(
        skm.f1_score(labels, (score >= expected).astype(int), zero_division=0)
    )


def test_the_best_threshold_beats_the_default_half(scored) -> None:
    # The whole reason this function exists: 0.5 is right only when the classes are balanced
    # and both error types cost the same.
    labels, score, ds = scored
    best = best_threshold(ds, "y", "s", thresholds=100)["f1"]
    assert best >= skm.f1_score(labels, (score >= 0.5).astype(int), zero_division=0)


def test_a_recall_weighted_objective_moves_the_threshold_down(scored) -> None:
    _, _, ds = scored
    balanced = best_threshold(ds, "y", "s", objective="f1", thresholds=100)["threshold"]
    recall_heavy = best_threshold(ds, "y", "s", objective="fbeta", beta=4.0, thresholds=100)[
        "threshold"
    ]
    assert recall_heavy <= balanced


def test_youden_is_available_as_an_objective(scored) -> None:
    _, _, ds = scored
    assert 0.0 <= best_threshold(ds, "y", "s", objective="youden")["threshold"] <= 1.0


def test_best_threshold_rejects_an_unknown_objective(scored) -> None:
    _, _, ds = scored
    with pytest.raises(PlanError, match="objective must be"):
        best_threshold(ds, "y", "s", objective="auc")


def test_the_cost_optimal_threshold_costs_no_more_than_any_other(scored) -> None:
    _, _, ds = scored
    curve = expected_cost_curve(
        ds, "y", "s", cost_false_positive=1.0, cost_false_negative=10.0, thresholds=100
    ).to_pydict()
    best = best_cost_threshold(
        ds, "y", "s", cost_false_positive=1.0, cost_false_negative=10.0, thresholds=100
    )
    assert best["total_cost"] == pytest.approx(min(curve["total_cost"]))


def test_asymmetric_costs_beat_the_f1_optimal_cutoff(scored) -> None:
    # When a miss costs ten times a false alarm, optimizing F1 leaves real money on the
    # table — F1 implicitly assumes the two cost the same.
    _, _, ds = scored
    costs = {"cost_false_positive": 1.0, "cost_false_negative": 10.0}
    curve = expected_cost_curve(ds, "y", "s", thresholds=100, **costs).to_pydict()
    by_threshold = dict(zip(curve["threshold"], curve["total_cost"], strict=True))
    f1_cutoff = best_threshold(ds, "y", "s", thresholds=100)["threshold"]
    optimal = best_cost_threshold(ds, "y", "s", thresholds=100, **costs)
    assert optimal["total_cost"] <= by_threshold[f1_cutoff]


def test_a_cost_curve_is_ordered_cheapest_first(scored) -> None:
    _, _, ds = scored
    curve = expected_cost_curve(
        ds, "y", "s", cost_false_positive=2.0, cost_false_negative=3.0, thresholds=20
    ).to_pydict()
    assert curve["total_cost"] == sorted(curve["total_cost"])


def test_a_cost_curve_rejects_a_negative_cost(scored) -> None:
    _, _, ds = scored
    with pytest.raises(PlanError, match="non-negative"):
        expected_cost_curve(ds, "y", "s", cost_false_positive=-1.0, cost_false_negative=1.0)


# --- comparing models -------------------------------------------------------------------------


def test_compare_models_ranks_the_better_model_first(scored) -> None:
    labels, score, _ = scored
    ds = bt.from_pydict(
        {"y": labels.tolist(), "good": score.tolist(), "inverted": (1 - score).tolist()}
    )
    got = compare_models(ds, "y", {"good": "good", "inverted": "inverted"}, metrics=["accuracy"])
    ranked = got.sort("accuracy", descending=True).to_pydict()
    assert ranked["model"][0] == "good"


def test_compare_models_agrees_with_evaluating_each_model_alone(scored) -> None:
    labels, score, _ = scored
    ds = bt.from_pydict({"y": labels.tolist(), "a": score.tolist()})
    together = compare_models(ds, "y", {"a": "a"}, metrics=["accuracy", "f1"]).to_pydict()
    alone = ds.ml.evaluate("y", y_score="a", metrics=["accuracy", "f1"])
    assert together["accuracy"][0] == pytest.approx(alone["accuracy"])
    assert together["f1"][0] == pytest.approx(alone["f1"])


def test_compare_models_handles_hard_predictions() -> None:
    ds = bt.from_pydict({"y": [1, 0, 1, 0], "a": [1, 0, 1, 0], "b": [0, 1, 0, 1]})
    got = compare_models(
        ds, "y", {"a": "a", "b": "b"}, task="binary", metrics=["accuracy"], scores=False
    ).to_pydict()
    assert dict(zip(got["model"], got["accuracy"], strict=True)) == {"a": 1.0, "b": 0.0}


def test_compare_models_refuses_a_rank_metric(scored) -> None:
    _, _, ds = scored
    with pytest.raises(PlanError, match="sort per model"):
        compare_models(ds, "y", {"m": "s"}, metrics=["roc_auc"])


def test_compare_models_needs_a_model(scored) -> None:
    _, _, ds = scored
    with pytest.raises(PlanError, match="at least one model"):
        compare_models(ds, "y", {})


def test_compare_models_names_a_missing_prediction_column(scored) -> None:
    _, _, ds = scored
    with pytest.raises(ColumnNotFoundError):
        compare_models(ds, "y", {"m": "nope"})


# --- the feature profile ---------------------------------------------------------------------


def test_feature_profile_suggests_dropping_a_constant_column() -> None:
    ds = bt.from_pydict({"flat": [1.0] * 20, "fine": [float(i) for i in range(20)]})
    got = feature_profile(ds).sort("column").to_pydict()
    assert dict(zip(got["column"], got["suggestion"], strict=True))["flat"] == "drop"


def test_feature_profile_suggests_a_power_transform_for_a_skewed_column() -> None:
    ds = bt.from_pydict({"revenue": [float(2**i) for i in range(14)]})
    assert feature_profile(ds).to_pydict()["suggestion"] == ["power transform"]


def test_feature_profile_suggests_imputing_a_mostly_missing_column() -> None:
    ds = bt.from_pydict({"sparse": [1.0, 2.0, *([None] * 8)]})
    assert feature_profile(ds).to_pydict()["suggestion"] == ["impute"]


def test_feature_profile_leaves_a_well_behaved_column_alone() -> None:
    rng = np.random.default_rng(12)
    ds = bt.from_pydict({"x": rng.normal(size=500).tolist()})
    assert feature_profile(ds).to_pydict()["suggestion"] == ["ready"]


def test_feature_profile_reports_the_shape_numbers() -> None:
    ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0]})
    got = feature_profile(ds).to_pydict()
    assert set(got) == {
        "column",
        "null_rate",
        "mode_share",
        "skew",
        "excess_kurtosis",
        "robust_cv",
        "suggestion",
    }
