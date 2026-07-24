"""Statistical analysis, drift monitoring, and cross-validation splits.

Three groups, each pinned against an independent oracle where one exists — SciPy for the
statistics, a hand-computed value where SciPy's convention differs, and an invariant where
neither applies.

The invariants matter more than the point values here, because these are the functions a
pipeline trusts silently: a fold assignment that is not disjoint leaks the validation set
into training, a group split that lets a group span folds inflates every score, and a
contingency table missing its empty cells halves a chi-squared statistic without erroring.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError
from batcher.ml.splitting import (
    fold_column,
    group_kfold,
    kfold,
    stratified_kfold,
    time_series_split,
)
from batcher.ml.stats import (
    anova_f,
    categorical_drift,
    chi_square,
    cramers_v,
    drift_report,
    entropy,
    gini_impurity,
    herfindahl_index,
    information_value,
    js_divergence,
    kl_divergence,
    median_abs_deviation,
    mode_share,
    mutual_information,
    outlier_mask,
    population_stability_index,
    spearman_corr,
    trimmed_mean,
    winsorized_mean,
    woe_table,
)

pytestmark = pytest.mark.unit

scipy_stats = pytest.importorskip("scipy.stats", reason="SciPy is the statistical oracle")


# --- rank and distribution statistics ------------------------------------------------


def test_spearman_matches_scipy() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=300)
    y = np.exp(x) + rng.normal(size=300) * 0.1
    ds = bt.from_pydict({"x": x.tolist(), "y": y.tolist()})
    assert spearman_corr(ds, "x", "y") == pytest.approx(scipy_stats.spearmanr(x, y).statistic)


def test_spearman_matches_scipy_with_heavy_ties() -> None:
    # Average ranks, not competition ranks: the two differ exactly here.
    rng = np.random.default_rng(2)
    x = np.round(rng.random(200), 1)
    y = np.round(rng.random(200), 1)
    ds = bt.from_pydict({"x": x.tolist(), "y": y.tolist()})
    assert spearman_corr(ds, "x", "y") == pytest.approx(scipy_stats.spearmanr(x, y).statistic)


def test_spearman_sees_a_monotone_relationship_pearson_underrates() -> None:
    x = np.arange(1.0, 51.0)
    ds = bt.from_pydict({"x": x.tolist(), "y": (x**4).tolist()})
    pearson = ds.agg(r=bt.corr("x", "y")).collect().column("r")[0].as_py()
    assert spearman_corr(ds, "x", "y") == pytest.approx(1.0)
    assert pearson < 0.95


def test_entropy_of_a_uniform_distribution_is_log2_k() -> None:
    ds = bt.from_pydict({"c": ["a", "b", "c", "d"]})
    assert entropy(ds, "c") == pytest.approx(2.0)


def test_entropy_of_a_constant_column_is_zero() -> None:
    ds = bt.from_pydict({"c": ["a"] * 10})
    assert entropy(ds, "c") == pytest.approx(0.0)


def test_entropy_matches_scipy() -> None:
    ds = bt.from_pydict({"c": ["a"] * 5 + ["b"] * 3 + ["c"] * 2})
    expected = scipy_stats.entropy([0.5, 0.3, 0.2], base=2)
    assert entropy(ds, "c") == pytest.approx(expected)


def test_gini_impurity_and_herfindahl_are_complements() -> None:
    ds = bt.from_pydict({"c": ["a"] * 7 + ["b"] * 3})
    assert gini_impurity(ds, "c") + herfindahl_index(ds, "c") == pytest.approx(1.0)


def test_mode_share_flags_a_column_that_is_really_a_flag() -> None:
    ds = bt.from_pydict({"x": [0] * 94 + list(range(6))})
    assert mode_share(ds, "x") > 0.9


# --- association ---------------------------------------------------------------------


def test_chi_square_counts_the_empty_cells_too() -> None:
    # The bug this pins: summing only the cells a group_by produced halved the statistic on
    # a perfectly associated table, which reads as "no relationship".
    ds = bt.from_pydict({"a": ["x", "x", "y", "y"], "b": ["p", "p", "q", "q"]})
    expected = scipy_stats.chi2_contingency([[2, 0], [0, 2]], correction=False).statistic
    assert chi_square(ds, "a", "b") == pytest.approx(expected)


def test_chi_square_matches_scipy_on_a_larger_table() -> None:
    rng = np.random.default_rng(7)
    left = rng.integers(0, 3, size=400)
    right = (left + rng.integers(0, 2, size=400)) % 3
    ds = bt.from_pydict({"a": left.tolist(), "b": right.tolist()})
    table = np.zeros((3, 3))
    for i, j in zip(left, right, strict=True):
        table[i, j] += 1
    expected = scipy_stats.chi2_contingency(table, correction=False).statistic
    assert chi_square(ds, "a", "b") == pytest.approx(expected)


def test_chi_square_of_independent_columns_is_zero() -> None:
    ds = bt.from_pydict({"a": ["x", "x", "y", "y"], "b": ["p", "q", "p", "q"]})
    assert chi_square(ds, "a", "b") == pytest.approx(0.0)


def test_cramers_v_is_one_for_a_perfect_association() -> None:
    ds = bt.from_pydict({"a": ["x", "x", "y", "y"], "b": ["p", "p", "q", "q"]})
    assert cramers_v(ds, "a", "b") == pytest.approx(1.0)


def test_cramers_v_does_not_grow_with_the_row_count() -> None:
    # Unlike chi-squared, which is why it is the number to rank features by.
    small = bt.from_pydict({"a": ["x", "y"] * 50, "b": ["p", "q"] * 50})
    large = bt.from_pydict({"a": ["x", "y"] * 5000, "b": ["p", "q"] * 5000})
    assert cramers_v(small, "a", "b") == pytest.approx(cramers_v(large, "a", "b"))


def test_mutual_information_equals_the_entropy_when_one_column_determines_the_other() -> None:
    ds = bt.from_pydict({"a": ["x", "x", "y", "y"], "b": ["p", "p", "q", "q"]})
    assert mutual_information(ds, "a", "b") == pytest.approx(entropy(ds, "a"))


def test_mutual_information_of_independent_columns_is_zero() -> None:
    ds = bt.from_pydict({"a": ["x", "x", "y", "y"], "b": ["p", "q", "p", "q"]})
    assert mutual_information(ds, "a", "b") == pytest.approx(0.0, abs=1e-12)


def test_anova_f_matches_scipy() -> None:
    rng = np.random.default_rng(4)
    first = rng.normal(0.0, 1.0, size=60)
    second = rng.normal(1.5, 1.0, size=60)
    third = rng.normal(3.0, 1.0, size=60)
    ds = bt.from_pydict(
        {
            "v": [*first.tolist(), *second.tolist(), *third.tolist()],
            "g": ["a"] * 60 + ["b"] * 60 + ["c"] * 60,
        }
    )
    expected = scipy_stats.f_oneway(first, second, third).statistic
    assert anova_f(ds, "v", "g") == pytest.approx(expected, rel=1e-9)


# --- robust location and spread -------------------------------------------------------


def test_trimmed_mean_ignores_a_corrupted_tail() -> None:
    clean = [float(i) for i in range(1, 101)]
    ds = bt.from_pydict({"x": [*clean, 1e12, -1e12]})
    assert trimmed_mean(ds, "x", proportion=0.1) == pytest.approx(50.5, rel=0.05)


def test_trimmed_mean_rejects_an_impossible_proportion() -> None:
    ds = bt.from_pydict({"x": [1.0]})
    with pytest.raises(PlanError, match="proportion"):
        trimmed_mean(ds, "x", proportion=0.5)


def test_winsorized_mean_keeps_the_row_count() -> None:
    values = [float(i) for i in range(1, 101)]
    ds = bt.from_pydict({"x": [*values, 1e12]})
    # Clamping rather than dropping: the extreme row is pulled in, not removed.
    assert winsorized_mean(ds, "x", proportion=0.1) < 200.0


def test_median_abs_deviation_matches_scipy() -> None:
    rng = np.random.default_rng(5)
    values = rng.normal(size=400)
    ds = bt.from_pydict({"x": values.tolist()})
    expected = scipy_stats.median_abs_deviation(values, scale="normal")
    # The engine's median interpolates between the two middle order statistics where NumPy
    # averages them, which differs in the last few digits on an even row count.
    assert median_abs_deviation(ds, "x") == pytest.approx(expected, rel=1e-5)


def test_median_abs_deviation_survives_half_the_data_being_garbage() -> None:
    ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 1e9, 1e9, 1e9]})
    assert median_abs_deviation(ds, "x", scale=1.0) < 1e8


def test_outlier_mask_selects_the_extreme_rows() -> None:
    ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 1000.0]})
    spread = median_abs_deviation(ds, "x")
    kept = ds.filter(outlier_mask("x", center=3.0, spread=spread)).to_pydict()["x"]
    assert kept == [1000.0]


def test_outlier_mask_with_no_spread_falls_back_to_inequality() -> None:
    ds = bt.from_pydict({"x": [5.0, 5.0, 9.0]})
    kept = ds.filter(outlier_mask("x", center=5.0, spread=0.0)).to_pydict()["x"]
    assert kept == [9.0]


# --- drift ---------------------------------------------------------------------------


def test_psi_is_zero_for_identical_distributions() -> None:
    values = [float(i) for i in range(500)]
    reference = bt.from_pydict({"x": values})
    current = bt.from_pydict({"x": values})
    assert population_stability_index(reference, current, "x") == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_the_size_of_the_shift() -> None:
    values = [float(i) for i in range(500)]
    reference = bt.from_pydict({"x": values})
    small = bt.from_pydict({"x": [v + 20 for v in values]})
    large = bt.from_pydict({"x": [v + 400 for v in values]})
    assert population_stability_index(reference, small, "x") < population_stability_index(
        reference, large, "x"
    )


def test_psi_crosses_the_conventional_alarm_threshold_on_a_real_shift() -> None:
    values = [float(i) for i in range(500)]
    reference = bt.from_pydict({"x": values})
    shifted = bt.from_pydict({"x": [v + 250 for v in values]})
    assert population_stability_index(reference, shifted, "x") > 0.25


def test_psi_flags_the_same_drift_from_either_direction() -> None:
    # PSI's formula is symmetric, but the *binning* is not: the edges come from whichever
    # dataset is passed as the reference, so the two directions differ in magnitude on a
    # scale change. What must hold either way is the decision: both flag drift.
    first = bt.from_pydict({"x": [float(i) for i in range(200)]})
    second = bt.from_pydict({"x": [float(i) * 2 for i in range(200)]})
    forward = population_stability_index(first, second, "x", buckets=4)
    backward = population_stability_index(second, first, "x", buckets=4)
    assert forward > 0.25 and backward > 0.25


def test_kl_divergence_is_zero_for_identical_distributions() -> None:
    values = [float(i) for i in range(200)]
    a = bt.from_pydict({"x": values})
    b = bt.from_pydict({"x": values})
    assert kl_divergence(a, b, "x") == pytest.approx(0.0, abs=1e-9)


def test_js_divergence_is_bounded_by_one_bit() -> None:
    reference = bt.from_pydict({"x": [float(i) for i in range(200)]})
    shifted = bt.from_pydict({"x": [float(i) + 1000 for i in range(200)]})
    assert 0.0 <= js_divergence(reference, shifted, "x") <= 1.0


def test_drift_on_a_constant_reference_column_is_an_actionable_error() -> None:
    reference = bt.from_pydict({"x": [1.0] * 10})
    current = bt.from_pydict({"x": [2.0] * 10})
    with pytest.raises(PlanError, match="is constant"):
        population_stability_index(reference, current, "x")


def test_categorical_drift_reads_as_the_share_of_mass_that_moved() -> None:
    reference = bt.from_pydict({"c": ["a", "a", "b", "b"]})
    current = bt.from_pydict({"c": ["a", "a", "a", "b"]})
    assert categorical_drift(reference, current, "c") == pytest.approx(0.25)


def test_categorical_drift_handles_a_category_present_on_only_one_side() -> None:
    reference = bt.from_pydict({"c": ["a", "a"]})
    current = bt.from_pydict({"c": ["a", "z"]})
    assert categorical_drift(reference, current, "c") == pytest.approx(0.5)


def test_drift_report_ranks_the_most_drifted_column_first() -> None:
    reference = bt.from_pydict(
        {"stable": [float(i) for i in range(200)], "moved": [float(i) for i in range(200)]}
    )
    current = bt.from_pydict(
        {
            "stable": [float(i) for i in range(200)],
            "moved": [float(i) + 500 for i in range(200)],
        }
    )
    report = drift_report(reference, current, ["stable", "moved"], buckets=5).to_pydict()
    assert report["column"][0] == "moved"
    assert report["mean_shift"][0] == pytest.approx(500.0)


def test_drift_report_needs_a_column() -> None:
    ds = bt.from_pydict({"x": [1.0]})
    with pytest.raises(PlanError, match="at least one column"):
        drift_report(ds, ds, [])


# --- weight of evidence ---------------------------------------------------------------


def test_woe_is_monotone_for_a_monotone_feature() -> None:
    rng = np.random.default_rng(8)
    feature = rng.random(2000)
    label = (rng.random(2000) < feature).astype(int)
    ds = bt.from_pydict({"x": feature.tolist(), "y": label.tolist()})
    table = woe_table(ds, "x", "y", buckets=5).to_pydict()
    assert table["woe"] == sorted(table["woe"])


def test_information_value_is_large_for_a_predictive_feature() -> None:
    rng = np.random.default_rng(9)
    feature = rng.random(2000)
    label = (rng.random(2000) < feature).astype(int)
    ds = bt.from_pydict({"x": feature.tolist(), "y": label.tolist()})
    assert information_value(ds, "x", "y", buckets=10) > 0.3


def test_information_value_is_near_zero_for_a_useless_feature() -> None:
    rng = np.random.default_rng(10)
    ds = bt.from_pydict({"x": rng.random(2000).tolist(), "y": rng.integers(0, 2, 2000).tolist()})
    assert information_value(ds, "x", "y", buckets=10) < 0.05


def test_woe_table_positive_rate_and_counts_agree() -> None:
    rng = np.random.default_rng(11)
    feature = rng.random(500)
    label = (rng.random(500) < feature).astype(int)
    ds = bt.from_pydict({"x": feature.tolist(), "y": label.tolist()})
    table = woe_table(ds, "x", "y", buckets=5).to_pydict()
    for rows, positives, rate in zip(
        table["rows"], table["positives"], table["positive_rate"], strict=True
    ):
        assert rate == pytest.approx(positives / rows)


# --- cross-validation splits ----------------------------------------------------------


def test_kfold_partitions_every_row_exactly_once() -> None:
    ds = bt.range(0, 1000)
    folds = kfold(ds, 5, key="value")
    assert sum(validate.count() for _, validate in folds) == 1000


def test_kfold_train_and_validation_are_disjoint() -> None:
    ds = bt.range(0, 500)
    for train, validate in kfold(ds, 4, key="value"):
        assert train.count() + validate.count() == 500
        overlap = train.select("value").intersect(validate.select("value"))
        assert overlap.count() == 0


def test_kfold_is_reproducible_for_the_same_seed() -> None:
    ds = bt.range(0, 300)
    first = [v.count() for _, v in kfold(ds, 3, seed=7, key="value")]
    second = [v.count() for _, v in kfold(ds, 3, seed=7, key="value")]
    assert first == second


def test_a_different_seed_gives_a_different_split() -> None:
    ds = bt.range(0, 300)
    first = sorted(kfold(ds, 3, seed=1, key="value")[0][1].to_pydict()["value"])
    second = sorted(kfold(ds, 3, seed=2, key="value")[0][1].to_pydict()["value"])
    assert first != second


def test_fold_column_uses_every_fold() -> None:
    folded = fold_column(bt.range(0, 1000), 5, key="value")
    assert sorted(set(folded.to_pydict()["fold"])) == [0, 1, 2, 3, 4]


def test_kfold_rejects_a_single_fold() -> None:
    with pytest.raises(PlanError, match="at least 2 folds"):
        kfold(bt.range(0, 10), 1)


def test_stratified_kfold_preserves_the_label_ratio() -> None:
    ds = bt.from_pydict({"y": [0] * 900 + [1] * 100, "x": list(range(1000))})
    folds = stratified_kfold(ds, "y", 5, key="x")
    positives = [v.filter(bt.col("y") == 1).count() for _, v in folds]
    assert positives == [20, 20, 20, 20, 20]


def test_stratified_kfold_still_partitions_every_row() -> None:
    ds = bt.from_pydict({"y": [0] * 90 + [1] * 10, "x": list(range(100))})
    folds = stratified_kfold(ds, "y", 5, key="x")
    assert sum(v.count() for _, v in folds) == 100


def test_stratified_kfold_names_a_missing_label_column() -> None:
    ds = bt.from_pydict({"x": [1, 2, 3]})
    with pytest.raises(ColumnNotFoundError):
        stratified_kfold(ds, "nope", 2)


def test_group_kfold_never_lets_a_group_span_folds() -> None:
    # The leak this prevents: the same entity in train and validation makes the model
    # memorize it, cross-validation looks excellent, and production does not.
    users = [f"u{i // 20}" for i in range(1000)]
    ds = bt.from_pydict({"user": users, "x": list(range(1000))})
    folds = group_kfold(ds, "user", 5)
    for train, validate in folds:
        train_users = set(train.select("user").distinct().to_pydict()["user"])
        validate_users = set(validate.select("user").distinct().to_pydict()["user"])
        assert not (train_users & validate_users)


def test_time_series_split_never_trains_on_the_future() -> None:
    ds = bt.from_pydict({"t": list(range(200)), "x": list(range(200))})
    for train, validate in ds.ml.time_series_split("t", 4):
        if train.count() and validate.count():
            assert max(train.to_pydict()["t"]) < min(validate.to_pydict()["t"])


def test_time_series_split_expands_the_training_window() -> None:
    ds = bt.from_pydict({"t": list(range(200)), "x": list(range(200))})
    sizes = [train.count() for train, _ in time_series_split(ds, "t", 4)]
    assert sizes == sorted(sizes)


def test_a_rolling_time_series_split_keeps_the_window_bounded() -> None:
    ds = bt.from_pydict({"t": list(range(200)), "x": list(range(200))})
    sizes = [train.count() for train, _ in time_series_split(ds, "t", 4, expanding=False)]
    assert max(sizes[1:]) <= sizes[0] + 1


def test_time_series_split_rejects_a_zero_split_count() -> None:
    ds = bt.from_pydict({"t": [1, 2, 3]})
    with pytest.raises(PlanError, match="at least 1 split"):
        time_series_split(ds, "t", 0)


def test_the_accessor_rejects_stratify_and_group_together() -> None:
    ds = bt.from_pydict({"y": [0, 1], "g": ["a", "b"]})
    with pytest.raises(PlanError, match="not both"):
        ds.ml.kfold(2, stratify="y", group="g")


def test_entropy_in_nats_matches_the_base_change() -> None:
    ds = bt.from_pydict({"c": ["a", "b", "c", "d"]})
    assert entropy(ds, "c", base=math.e) == pytest.approx(math.log(4))
