"""Statistical analysis expressions — robust spread, distribution shape, association, tests.

The data-science half of the statistical surface. `plan.functions.statistics` holds the
descriptive aggregates SQL itself defines (``var_pop``, ``stddev_pop``, ``geometric_mean``,
…); this package holds what an analyst reaches for *after* those: quantile-based location
and spread that a single outlier cannot move, shape and normality diagnostics, association
measures beyond Pearson, and the two-sample statistics behind an A/B test.

Everything here is an expression over the existing mergeable aggregates, so each one is a
single pass, works inside `group_by`, and runs distributed unchanged. Nothing here is a
p-value: see `inference` for why.
"""

from __future__ import annotations

from batcher.plan.functions.analysis.association import (
    correlation_ratio,
    point_biserial,
    signal_ratio,
)
from batcher.plan.functions.analysis.dispersion import (
    decile_ratio,
    interdecile_range,
    midhinge,
    quartile_dispersion,
    robust_cv,
    trimean,
)
from batcher.plan.functions.analysis.inference import (
    cohens_d,
    group_mean,
    hedges_g,
    mean_ci_half_width,
    proportion_ci_half_width,
    proportion_z_statistic,
    welch_df,
    welch_t_statistic,
)
from batcher.plan.functions.analysis.moments import (
    geometric_std,
    index_of_dispersion,
    relative_range,
    signal_to_noise,
    studentized_range,
)
from batcher.plan.functions.analysis.shape import (
    bowley_skew,
    jarque_bera,
    moors_kurtosis,
    pearson_mode_skew,
)
from batcher.plan.functions.analysis.weighted import (
    weighted_correlation,
    weighted_covariance,
    weighted_std,
    weighted_var,
)

__all__ = [
    "bowley_skew",
    "cohens_d",
    "correlation_ratio",
    "decile_ratio",
    "geometric_std",
    "group_mean",
    "hedges_g",
    "index_of_dispersion",
    "interdecile_range",
    "jarque_bera",
    "mean_ci_half_width",
    "midhinge",
    "moors_kurtosis",
    "pearson_mode_skew",
    "point_biserial",
    "proportion_ci_half_width",
    "proportion_z_statistic",
    "quartile_dispersion",
    "relative_range",
    "robust_cv",
    "signal_ratio",
    "signal_to_noise",
    "studentized_range",
    "trimean",
    "weighted_correlation",
    "weighted_covariance",
    "weighted_std",
    "weighted_var",
    "welch_df",
    "welch_t_statistic",
]
