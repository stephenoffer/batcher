"""Statistical analysis and drift monitoring over a `Dataset`.

Five modules, split by the question each answers. `descriptive` summarizes one column's
distribution (entropy, concentration, rank correlation); `association` measures how two
columns relate (chi-squared, Cramer's V, mutual information, ANOVA); `robust` gives
location and spread a corrupted tail cannot move; `drift` compares a reference dataset
against a current one (PSI, divergences, weight of evidence, a per-column report);
`hypothesis` pairs a test statistic with the p-value that turns it into a decision.

Every one of them is expressed in relational operators — a window, a `group_by`, a second
aggregate over the first — so nothing materializes the column on the driver.

The single-pass statistical *expressions* are elsewhere and reachable as ``bt.trimean`` /
``bt.welch_t_statistic``; see `batcher.plan.functions.analysis`.
"""

from __future__ import annotations

from batcher.ml.stats.association import (
    anova_f,
    chi_square,
    cohens_f,
    cramers_v,
    epsilon_squared,
    eta_squared,
    mutual_information,
    omega_squared,
    theils_u,
)
from batcher.ml.stats.descriptive import (
    entropy,
    gini_impurity,
    herfindahl_index,
    mode_share,
    normalized_entropy,
    spearman_corr,
)
from batcher.ml.stats.drift import (
    categorical_drift,
    drift_report,
    information_value,
    js_divergence,
    kl_divergence,
    population_stability_index,
    woe_table,
)
from batcher.ml.stats.homogeneity import bartlett_test, levene_test
from batcher.ml.stats.hypothesis import (
    TestResult,
    anova_test,
    binomial_test,
    chi_square_test,
    mcnemar_test,
    normality_test,
    pearson_test,
    proportion_ztest,
    spearman_test,
    t_test_1samp,
    t_test_ind,
)
from batcher.ml.stats.multivariate import (
    correlation_matrix,
    covariance_matrix,
    partial_correlation,
    variance_inflation_factor,
)
from batcher.ml.stats.nonparametric import (
    cliffs_delta,
    common_language_effect_size,
    friedman_test,
    kruskal_wallis,
    mann_whitney_u,
    wilcoxon_signed_rank,
)
from batcher.ml.stats.robust import (
    mean_abs_deviation,
    median_abs_deviation,
    outlier_mask,
    trimmed_mean,
    winsorized_mean,
)

__all__ = [
    "TestResult",
    "anova_f",
    "anova_test",
    "bartlett_test",
    "binomial_test",
    "categorical_drift",
    "chi_square",
    "chi_square_test",
    "cliffs_delta",
    "cohens_f",
    "common_language_effect_size",
    "correlation_matrix",
    "covariance_matrix",
    "cramers_v",
    "drift_report",
    "entropy",
    "epsilon_squared",
    "eta_squared",
    "friedman_test",
    "gini_impurity",
    "herfindahl_index",
    "information_value",
    "js_divergence",
    "kl_divergence",
    "kruskal_wallis",
    "levene_test",
    "mann_whitney_u",
    "mcnemar_test",
    "mean_abs_deviation",
    "median_abs_deviation",
    "mode_share",
    "mutual_information",
    "normality_test",
    "normalized_entropy",
    "omega_squared",
    "outlier_mask",
    "partial_correlation",
    "pearson_test",
    "population_stability_index",
    "proportion_ztest",
    "spearman_corr",
    "spearman_test",
    "t_test_1samp",
    "t_test_ind",
    "theils_u",
    "trimmed_mean",
    "variance_inflation_factor",
    "wilcoxon_signed_rank",
    "winsorized_mean",
    "woe_table",
]
