"""Model-evaluation metrics — parity with scikit-learn, and the per-group query.

Every metric here is checked against `sklearn.metrics` on the same data, because "our
definition of F1" is not a thing anyone should have to read code to discover. The
tolerance is exact equality wherever the two compute the same closed form, which is most
of them; the few that differ do so only in the last bits of floating-point association.

The second half pins what scikit-learn cannot do at all: the same metrics computed *per
group* in one pass, and the diagnostic tables as lazy Datasets.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError
from batcher.ml.metrics import (
    METRIC_SETS,
    average_precision,
    calibration_curve,
    confusion_matrix,
    evaluate,
    gini_coefficient,
    ks_statistic,
    lift_table,
    roc_auc,
    threshold_sweep,
)

pytestmark = pytest.mark.unit

skm = pytest.importorskip("sklearn.metrics", reason="scikit-learn is the metric oracle")

EXACT = 1e-12


@pytest.fixture(scope="module")
def regression() -> tuple[np.ndarray, np.ndarray, bt.Dataset]:
    """A regression problem with a large-magnitude target (where naive formulas lose precision)."""
    rng = np.random.default_rng(3)
    actual = rng.normal(size=500) * 10 + 50
    predicted = actual + rng.normal(size=500) * 2
    ds = bt.from_pydict({"y": actual.tolist(), "p": predicted.tolist()})
    return actual, predicted, ds


@pytest.fixture(scope="module")
def binary() -> tuple[np.ndarray, np.ndarray, np.ndarray, bt.Dataset]:
    """A binary problem with a genuinely informative score (so AUC is not degenerate)."""
    rng = np.random.default_rng(11)
    score = rng.random(600)
    labels = (rng.random(600) < score).astype(int)
    hard = (score > 0.5).astype(int)
    ds = bt.from_pydict({"y": labels.tolist(), "p": hard.tolist(), "s": score.tolist()})
    return labels, hard, score, ds


def _one(ds: bt.Dataset, **aggs: object) -> dict[str, float]:
    """Run a one-row aggregate and return it as a plain dict."""
    row = ds.agg(**aggs).collect()
    return {name: row.column(name)[0].as_py() for name in row.column_names}


# --- regression parity --------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "builder", "oracle"),
    [
        ("mse", bt.mse, skm.mean_squared_error),
        ("rmse", bt.rmse, skm.root_mean_squared_error),
        ("mae", bt.mae, skm.mean_absolute_error),
        ("r2", bt.r2, skm.r2_score),
        ("explained_variance", bt.explained_variance, skm.explained_variance_score),
        ("mape", bt.mape, skm.mean_absolute_percentage_error),
        ("medae", bt.medae, skm.median_absolute_error),
        ("max_error", bt.max_error, skm.max_error),
    ],
)
def test_regression_metric_matches_sklearn(regression, name, builder, oracle) -> None:
    actual, predicted, ds = regression
    got = _one(ds, m=builder("y", "p"))["m"]
    assert got == pytest.approx(oracle(actual, predicted), abs=EXACT)


def test_msle_matches_sklearn() -> None:
    rng = np.random.default_rng(4)
    actual = rng.random(200) * 100
    predicted = actual * rng.uniform(0.8, 1.2, size=200)
    ds = bt.from_pydict({"y": actual.tolist(), "p": predicted.tolist()})
    got = _one(ds, m=bt.msle("y", "p"))["m"]
    assert got == pytest.approx(skm.mean_squared_log_error(actual, predicted), abs=EXACT)


def test_wape_is_the_ratio_of_totals() -> None:
    ds = bt.from_pydict({"y": [0.0, 100.0], "p": [1.0, 90.0]})
    assert _one(ds, m=bt.wape("y", "p"))["m"] == pytest.approx(11.0 / 100.0)


def test_smape_is_finite_when_the_actual_is_zero() -> None:
    ds = bt.from_pydict({"y": [0.0, 0.0], "p": [0.0, 4.0]})
    assert _one(ds, m=bt.smape("y", "p"))["m"] == pytest.approx(1.0)


def test_mape_excludes_zero_actuals_from_the_denominator_too() -> None:
    # The failure this pins: excluding a row from the numerator but not the denominator
    # silently halves the metric.
    ds = bt.from_pydict({"y": [0.0, 10.0], "p": [5.0, 11.0]})
    assert _one(ds, m=bt.mape("y", "p"))["m"] == pytest.approx(0.1)


def test_pinball_loss_at_the_median_is_half_the_mae(regression) -> None:
    _, _, ds = regression
    got = _one(ds, pin=bt.pinball_loss("y", "p", quantile=0.5), mae=bt.mae("y", "p"))
    assert got["pin"] == pytest.approx(got["mae"] / 2.0)


def test_pinball_loss_weights_under_prediction_by_the_quantile() -> None:
    ds = bt.from_pydict({"y": [10.0], "p": [8.0]})
    assert _one(ds, m=bt.pinball_loss("y", "p", quantile=0.9))["m"] == pytest.approx(1.8)


@pytest.mark.parametrize("quantile", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_pinball_loss_is_never_negative_in_its_domain(quantile: float) -> None:
    """A loss is non-negative by definition, and outside [0, 1] this one was not."""
    ds = bt.from_pydict({"y": [1.0, 2.0, 3.0], "p": [11.0, 12.0, 13.0]})
    assert _one(ds, m=bt.pinball_loss("y", "p", quantile=quantile))["m"] >= 0.0


@pytest.mark.parametrize("quantile", [-0.001, -0.5, 1.001, 1.5, 90.0, float("nan")])
def test_pinball_loss_rejects_a_quantile_outside_the_unit_interval(quantile: float) -> None:
    """Past 1 the two weights stop sharing a sign, so the "loss" goes negative.

    On a forecast that overshoots by 10, ``quantile=1.5`` scored -5.0 and the percentile typo
    ``quantile=90`` scored -890.0, against 1.0 for the correct 0.9 -- and anything minimizing
    the metric is driven away from the data. `quantile`, `approx_quantile` and
    `token_estimate_quantile` all already rejected their domain; this did not.
    """
    ds = bt.from_pydict({"y": [1.0, 2.0], "p": [11.0, 12.0]})
    with pytest.raises(PlanError, match=r"quantile must be in \[0, 1\]"):
        _one(ds, m=bt.pinball_loss("y", "p", quantile=quantile))


@pytest.mark.parametrize("quantile", ["0.9", True, None])
def test_pinball_loss_rejects_a_non_numeric_quantile(quantile) -> None:
    ds = bt.from_pydict({"y": [1.0], "p": [2.0]})
    with pytest.raises(PlanError, match="must be a number"):
        _one(ds, m=bt.pinball_loss("y", "p", quantile=quantile))


def test_huber_loss_is_quadratic_below_delta_and_linear_above() -> None:
    ds = bt.from_pydict({"y": [0.0, 0.0], "p": [0.5, 10.0]})
    # 0.5 * 0.5^2 = 0.125 below delta; 1.0 * (10 - 0.5) = 9.5 above; mean = 4.8125.
    assert _one(ds, m=bt.huber_loss("y", "p"))["m"] == pytest.approx(4.8125)


def test_mean_bias_shows_a_systematic_offset_rmse_hides() -> None:
    ds = bt.from_pydict({"y": [1.0, 2.0], "p": [2.0, 3.0]})
    assert _one(ds, m=bt.mean_bias("y", "p"))["m"] == pytest.approx(1.0)


def test_regression_metrics_ignore_a_row_with_a_null_on_either_side() -> None:
    ds = bt.from_pydict({"y": [1.0, 2.0, None], "p": [1.0, 4.0, 9.0]})
    assert _one(ds, m=bt.mse("y", "p"))["m"] == pytest.approx(2.0)


# --- classification parity ----------------------------------------------------------


@pytest.mark.parametrize(
    ("builder", "oracle"),
    [
        (bt.accuracy, skm.accuracy_score),
        (bt.precision, skm.precision_score),
        (bt.recall, skm.recall_score),
        (bt.f1_score, skm.f1_score),
        (bt.balanced_accuracy, skm.balanced_accuracy_score),
        (bt.matthews_corrcoef, skm.matthews_corrcoef),
        (bt.cohen_kappa, skm.cohen_kappa_score),
    ],
)
def test_label_metric_matches_sklearn(binary, builder, oracle) -> None:
    labels, hard, _, ds = binary
    assert _one(ds, m=builder("y", "p"))["m"] == pytest.approx(oracle(labels, hard), abs=EXACT)


def test_fbeta_matches_sklearn(binary) -> None:
    labels, hard, _, ds = binary
    got = _one(ds, m=bt.fbeta_score("y", "p", beta=2.0))["m"]
    assert got == pytest.approx(skm.fbeta_score(labels, hard, beta=2.0), abs=EXACT)


def test_log_loss_matches_sklearn(binary) -> None:
    labels, _, score, ds = binary
    assert _one(ds, m=bt.log_loss("y", "s"))["m"] == pytest.approx(
        skm.log_loss(labels, score), abs=EXACT
    )


def test_brier_score_matches_sklearn(binary) -> None:
    labels, _, score, ds = binary
    assert _one(ds, m=bt.brier_score("y", "s"))["m"] == pytest.approx(
        skm.brier_score_loss(labels, score), abs=EXACT
    )


def test_confusion_counts_add_up_to_the_row_count(binary) -> None:
    _, _, _, ds = binary
    got = _one(
        ds,
        tp=bt.true_positives("y", "p"),
        fp=bt.false_positives("y", "p"),
        fn=bt.false_negatives("y", "p"),
        tn=bt.true_negatives("y", "p"),
    )
    assert sum(got.values()) == ds.count()


def test_specificity_and_false_positive_rate_are_complements(binary) -> None:
    _, _, _, ds = binary
    got = _one(ds, spec=bt.specificity("y", "p"), fpr=bt.false_positive_rate("y", "p"))
    assert got["spec"] + got["fpr"] == pytest.approx(1.0)


def test_a_string_label_column_works_with_an_explicit_positive_class() -> None:
    ds = bt.from_pydict({"y": ["churn", "stay", "churn"], "p": ["churn", "stay", "stay"]})
    got = _one(ds, m=bt.recall("y", "p", positive="churn"))["m"]
    assert got == pytest.approx(0.5)


def test_matthews_is_zero_rather_than_nan_when_a_denominator_vanishes() -> None:
    # Every row predicted positive: (tn + fp) * (tn + fn) is 0, and NaN here would poison
    # a whole evaluation report.
    ds = bt.from_pydict({"y": [1, 1], "p": [1, 1]})
    assert _one(ds, m=bt.matthews_corrcoef("y", "p"))["m"] == 0.0


def test_prevalence_is_the_base_rate() -> None:
    ds = bt.from_pydict({"y": [1, 0, 0, 0]})
    assert _one(ds, m=bt.prevalence("y"))["m"] == pytest.approx(0.25)


# --- rank-based metrics -------------------------------------------------------------


def test_roc_auc_matches_sklearn(binary) -> None:
    labels, _, score, ds = binary
    assert roc_auc(ds, "y", "s") == pytest.approx(skm.roc_auc_score(labels, score), abs=EXACT)


def test_roc_auc_is_exact_under_heavy_ties() -> None:
    # The rank identity is only correct with *average* ranks. A tie-heavy score column is
    # what catches an implementation that uses the competition rank alone.
    rng = np.random.default_rng(21)
    score = np.round(rng.random(400), 1)
    labels = (rng.random(400) < score).astype(int)
    ds = bt.from_pydict({"y": labels.tolist(), "s": score.tolist()})
    assert roc_auc(ds, "y", "s") == pytest.approx(skm.roc_auc_score(labels, score), abs=EXACT)


def test_average_precision_matches_sklearn(binary) -> None:
    labels, _, score, ds = binary
    got = average_precision(ds, "y", "s")
    assert got == pytest.approx(skm.average_precision_score(labels, score), abs=1e-9)


def test_gini_is_twice_the_auc_minus_one(binary) -> None:
    _, _, _, ds = binary
    assert gini_coefficient(ds, "y", "s") == pytest.approx(2 * roc_auc(ds, "y", "s") - 1)


def test_ks_statistic_is_one_for_a_perfect_separation() -> None:
    ds = bt.from_pydict({"y": [0, 0, 1, 1], "s": [0.1, 0.2, 0.8, 0.9]})
    assert ks_statistic(ds, "y", "s") == pytest.approx(1.0)


def test_ks_statistic_matches_the_max_cdf_gap(binary) -> None:
    labels, _, score, ds = binary
    order = np.argsort(score, kind="stable")
    sorted_labels = labels[order]
    positives = np.cumsum(sorted_labels) / labels.sum()
    negatives = np.cumsum(1 - sorted_labels) / (len(labels) - labels.sum())
    assert ks_statistic(ds, "y", "s") == pytest.approx(np.abs(positives - negatives).max())


def test_roc_auc_per_group_is_one_pass_over_the_whole_dataset(binary) -> None:
    labels, _, score, ds = binary
    rng = np.random.default_rng(2)
    groups = rng.integers(0, 3, size=len(labels))
    grouped = ds.with_columns(g=bt.lit(0)).to_pydict()
    frame = bt.from_pydict({**grouped, "g": groups.tolist()})
    per_group = roc_auc(frame, "y", "s", by="g").sort("g").to_pydict()
    for index, value in zip(per_group["g"], per_group["roc_auc"], strict=True):
        mask = groups == index
        assert value == pytest.approx(skm.roc_auc_score(labels[mask], score[mask]), abs=EXACT)


def test_a_rank_metric_names_the_missing_column() -> None:
    ds = bt.from_pydict({"y": [0, 1], "s": [0.1, 0.9]})
    with pytest.raises(ColumnNotFoundError):
        roc_auc(ds, "y", "nope")


# --- diagnostic tables --------------------------------------------------------------


def test_confusion_matrix_is_long_form_and_multi_class() -> None:
    ds = bt.from_pydict({"y": ["a", "b", "c", "a"], "p": ["a", "b", "a", "b"]})
    got = confusion_matrix(ds, "y", "p").sort("y", "p").to_pydict()
    assert list(zip(got["y"], got["p"], got["count"], strict=True)) == [
        ("a", "a", 1),
        ("a", "b", 1),
        ("b", "b", 1),
        ("c", "a", 1),
    ]


def test_threshold_sweep_precision_and_recall_move_the_right_way(binary) -> None:
    _, _, _, ds = binary
    swept = threshold_sweep(ds, "y", "s", thresholds=10).to_pydict()
    # Ordered by descending threshold, so recall increases as the cutoff falls.
    assert swept["recall"] == sorted(swept["recall"])
    assert swept["predicted_positive_rate"] == sorted(swept["predicted_positive_rate"])


def test_threshold_sweep_confusion_cells_sum_to_the_row_count(binary) -> None:
    _, _, _, ds = binary
    swept = threshold_sweep(ds, "y", "s", thresholds=8).to_pydict()
    total = ds.count()
    for cells in zip(swept["tp"], swept["fp"], swept["fn"], swept["tn"], strict=True):
        assert sum(cells) == pytest.approx(total)


def test_lift_table_top_bucket_beats_the_base_rate() -> None:
    ds = bt.from_pydict({"y": [1, 1, 1, 0, 0, 0], "s": [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]})
    table = lift_table(ds, "y", "s", buckets=2).to_pydict()
    assert table["lift"] == [2.0, 0.0]
    assert table["capture_rate"] == [1.0, 1.0]


def test_lift_table_cumulative_lift_ends_at_one() -> None:
    rng = np.random.default_rng(9)
    score = rng.random(200)
    labels = (rng.random(200) < score).astype(int)
    ds = bt.from_pydict({"y": labels.tolist(), "s": score.tolist()})
    table = lift_table(ds, "y", "s", buckets=5).to_pydict()
    assert table["cumulative_lift"][-1] == pytest.approx(1.0)


def test_calibration_curve_separates_a_miscalibrated_model() -> None:
    ds = bt.from_pydict({"y": [0, 0, 1, 1], "s": [0.05, 0.1, 0.9, 0.95]})
    curve = calibration_curve(ds, "y", "s", bins=2).to_pydict()
    assert curve["observed_rate"] == [0.0, 1.0]
    assert all(err < 0.1 for err in curve["calibration_error"])


def test_a_table_rejects_a_degenerate_bucket_count() -> None:
    ds = bt.from_pydict({"y": [0, 1], "s": [0.1, 0.9]})
    with pytest.raises(PlanError, match="at least 2"):
        lift_table(ds, "y", "s", buckets=1)


# --- evaluate -----------------------------------------------------------------------


def test_evaluate_binary_reports_the_whole_default_set(binary) -> None:
    _, _, _, ds = binary
    report = evaluate(ds, "y", y_score="s")
    assert list(report) == list(METRIC_SETS["binary"])


def test_evaluate_derives_hard_predictions_from_the_score(binary) -> None:
    labels, hard, _, ds = binary
    report = evaluate(ds, "y", y_score="s", metrics=["accuracy", "f1"])
    assert report["accuracy"] == pytest.approx(skm.accuracy_score(labels, hard))
    assert report["f1"] == pytest.approx(skm.f1_score(labels, hard))


def test_evaluate_honors_a_non_default_threshold(binary) -> None:
    labels, _, score, ds = binary
    report = evaluate(ds, "y", y_score="s", metrics=["accuracy"], threshold=0.8)
    expected = skm.accuracy_score(labels, (score >= 0.8).astype(int))
    assert report["accuracy"] == pytest.approx(expected)


def test_evaluate_regression_matches_the_individual_metrics(regression) -> None:
    actual, predicted, ds = regression
    report = evaluate(ds, "y", y_pred="p", task="regression")
    assert report["rmse"] == pytest.approx(skm.root_mean_squared_error(actual, predicted))
    assert report["r2"] == pytest.approx(skm.r2_score(actual, predicted))


def test_evaluate_by_group_returns_one_row_per_group() -> None:
    ds = bt.from_pydict(
        {
            "g": ["a", "a", "b", "b"],
            "y": [1, 0, 1, 0],
            "s": [0.9, 0.1, 0.2, 0.8],
        }
    )
    frame = evaluate(ds, "y", y_score="s", by="g", metrics=["accuracy", "roc_auc"]).sort("g")
    got = frame.to_pydict()
    assert got["g"] == ["a", "b"]
    assert got["roc_auc"] == [1.0, 0.0]


def test_evaluate_rejects_an_unknown_metric(binary) -> None:
    _, _, _, ds = binary
    with pytest.raises(PlanError, match="unknown metric"):
        evaluate(ds, "y", y_score="s", metrics=["f_one"])


def test_evaluate_rejects_an_unknown_task(binary) -> None:
    _, _, _, ds = binary
    with pytest.raises(PlanError, match="task must be"):
        evaluate(ds, "y", y_pred="p", task="clustering")


def test_evaluate_needs_a_prediction_of_some_kind() -> None:
    ds = bt.from_pydict({"y": [1, 0]})
    with pytest.raises(PlanError, match="y_pred"):
        evaluate(ds, "y")


def test_a_rank_metric_without_a_score_column_says_so(binary) -> None:
    _, _, _, ds = binary
    with pytest.raises(PlanError, match="y_score"):
        evaluate(ds, "y", y_pred="p", task="binary", metrics=["roc_auc"])


def test_the_dataset_accessor_reaches_the_same_evaluation(binary) -> None:
    _, _, _, ds = binary
    assert ds.ml.evaluate("y", y_score="s") == evaluate(ds, "y", y_score="s")


# --- the task the labels imply, not the one the argument list implied ------------------


def test_auto_scores_a_binary_classification_as_a_classification() -> None:
    """`evaluate("y", y_pred="p")` on 0/1 labels used to report RMSE, MAE and R2.

    `auto` answered from the *argument list* alone — a `y_score` meant binary, a `y_pred`
    meant regression, whatever the labels held. So the most ordinary call there is scored a
    classification as a regression: real numbers, computed correctly, answering a question
    nobody asked, with nothing in the result to say so.
    """
    ds = bt.from_pydict({"y": [1, 0, 1, 0], "p": [1, 0, 0, 1]})
    report = evaluate(ds, "y", y_pred="p")

    assert "accuracy" in report
    assert "rmse" not in report
    assert report["accuracy"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        ([1.0, 2.0, 3.0, 4.0], "rmse"),  # a float label is a regression however few values
        ([1, 0, 1, 0], "accuracy"),  # two integer classes -> binary
        ([True, False, True, False], "accuracy"),  # booleans too
        (["a", "b", "a", "b"], "accuracy"),  # and strings
        ([0, 1, 2, 1], "macro_f1"),  # three classes -> multiclass
    ],
)
def test_auto_reads_the_task_off_the_label_column(labels, expected) -> None:
    ds = bt.from_pydict({"y": labels, "p": labels})
    assert expected in evaluate(ds, "y", y_pred="p", positive=labels[0])


def test_a_high_cardinality_integer_label_stays_a_regression() -> None:
    """An id-like integer column is not a 200-class problem."""
    ds = bt.from_pydict({"y": list(range(200)), "p": list(range(200))})
    assert "rmse" in evaluate(ds, "y", y_pred="p")


def test_binary_without_a_score_reports_the_metrics_it_can() -> None:
    """Four of the ten binary defaults need a probability, so this raised outright.

    A metric the caller *names* and cannot have still raises, naming it. But a default set
    is not a request: with hard predictions alone the report is the six metrics that only
    need labels.
    """
    ds = bt.from_pydict({"y": [1, 0, 1, 0], "p": [1, 0, 0, 1]})
    report = evaluate(ds, "y", y_pred="p", task="binary")

    assert set(report) == {"accuracy", "precision", "recall", "f1", "balanced_accuracy", "mcc"}


def test_a_score_metric_the_caller_named_still_raises() -> None:
    ds = bt.from_pydict({"y": [1, 0, 1, 0], "p": [1, 0, 0, 1]})
    with pytest.raises(PlanError, match="needs y_score"):
        evaluate(ds, "y", y_pred="p", metrics=["roc_auc"])


def test_a_score_restores_the_full_binary_set() -> None:
    ds = bt.from_pydict({"y": [1, 0, 1, 0], "p": [1, 0, 0, 1], "s": [0.9, 0.1, 0.4, 0.6]})
    report = evaluate(ds, "y", y_pred="p", y_score="s", task="binary")
    assert set(report) == set(METRIC_SETS["binary"])


# --- tied scores ---------------------------------------------------------------------
#
# The `binary` fixture draws continuous scores, so every score above is distinct and the
# ranking metrics were only ever checked where ties cannot happen. Ties are not exotic:
# a clipped probability, a rounded score, a shallow tree and a calibrated output all
# produce them, and `average_precision` counted the positives per tie *group* against the
# rows per *row*, which is not a precision — on five distinct scores it returned values
# above 1.


@pytest.fixture
def tied() -> tuple[np.ndarray, np.ndarray, bt.Dataset]:
    """A binary problem whose scores are quantized to five levels, so ties are everywhere."""
    rng = np.random.default_rng(3)
    labels = (rng.random(400) > 0.55).astype(int)
    score = np.round(np.clip(labels * 0.3 + rng.random(400) * 0.9, 0, 1) * 4) / 4
    ds = bt.from_pydict({"y": labels.tolist(), "s": score.tolist()})
    return labels, score, ds


def test_tied_scores_really_are_tied(tied) -> None:
    """Guards the fixture itself: this test file's point is lost if the scores are distinct."""
    _, score, _ = tied
    assert len(set(score.tolist())) == 5


def test_average_precision_matches_sklearn_under_ties(tied) -> None:
    labels, score, ds = tied
    got = average_precision(ds, "y", "s")
    assert got == pytest.approx(skm.average_precision_score(labels, score), abs=1e-9)


def test_average_precision_stays_within_its_range_under_ties(tied) -> None:
    """The regression proper: this returned 1.35-1.59 when every row was its own denominator."""
    _, _, ds = tied
    assert 0.0 <= average_precision(ds, "y", "s") <= 1.0


def test_average_precision_of_a_single_tie_group_is_the_prevalence() -> None:
    """With one score for every row there is no ranking left, so AP is the positive rate."""
    ds = bt.from_pydict({"y": [0, 1, 0, 1, 1, 0], "s": [0.5] * 6})
    assert average_precision(ds, "y", "s") == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize(
    ("builder", "oracle"),
    [
        (roc_auc, lambda y, s: skm.roc_auc_score(y, s)),
        (average_precision, lambda y, s: skm.average_precision_score(y, s)),
        (gini_coefficient, lambda y, s: 2 * skm.roc_auc_score(y, s) - 1),
    ],
)
def test_ranking_metrics_match_sklearn_under_ties(tied, builder, oracle) -> None:
    labels, score, ds = tied
    assert builder(ds, "y", "s") == pytest.approx(oracle(labels, score), abs=1e-9)
