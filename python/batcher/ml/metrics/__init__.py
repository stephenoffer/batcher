"""Model evaluation over a `Dataset` — rank metrics, diagnostic tables, and `evaluate`.

The Dataset-level half of Batcher's metric surface. The other half is
`batcher.plan.functions.metrics`, whose metrics are plain `Expr` aggregates reachable as
``bt.rmse`` / ``bt.f1_score`` and usable inside any `agg()`.

What lives here is what cannot be an aggregate: the rank metrics (ROC AUC, average
precision, KS, Gini), which need a global ordering; the *ranking* metrics (precision@k,
NDCG, MRR), which are computed within a query and then averaged over queries; the
diagnostic tables (confusion matrix, threshold sweep, lift, calibration), which return a
`Dataset`; and `evaluate`, which runs a whole task's metric set in as few passes as the
metrics allow.
"""

from __future__ import annotations

from batcher.ml.metrics.calibration import (
    brier_skill_score,
    expected_calibration_error,
    maximum_calibration_error,
)
from batcher.ml.metrics.cluster_quality import (
    calinski_harabasz_score,
    davies_bouldin_score,
)
from batcher.ml.metrics.clustering import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    contingency_matrix,
    fowlkes_mallows_score,
    homogeneity_score,
    mutual_info_score,
    normalized_mutual_info_score,
    pair_confusion_matrix,
    rand_score,
    v_measure_score,
)
from batcher.ml.metrics.comparison import compare_models
from batcher.ml.metrics.evaluate import METRIC_SETS, evaluate, multiclass_averages
from batcher.ml.metrics.fairness import (
    demographic_parity_difference,
    disparate_impact_ratio,
    equal_opportunity_difference,
    equalized_odds_difference,
    group_fairness_report,
    predictive_parity_difference,
)
from batcher.ml.metrics.ranked import (
    average_precision,
    gini_coefficient,
    ks_statistic,
    roc_auc,
)
from batcher.ml.metrics.ranking import (
    hit_rate_at_k,
    map_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from batcher.ml.metrics.regression import (
    d2_absolute_error_score,
    d2_pinball_score,
    d2_tweedie_score,
    prediction_interval_coverage,
    residual_summary,
    top_k_accuracy,
)
from batcher.ml.metrics.tables import (
    calibration_curve,
    classification_report,
    confusion_matrix,
    lift_table,
    threshold_sweep,
)
from batcher.ml.metrics.thresholds import (
    best_cost_threshold,
    best_threshold,
    expected_cost_curve,
)

__all__ = [
    "METRIC_SETS",
    "adjusted_mutual_info_score",
    "adjusted_rand_score",
    "average_precision",
    "best_cost_threshold",
    "best_threshold",
    "brier_skill_score",
    "calibration_curve",
    "calinski_harabasz_score",
    "classification_report",
    "compare_models",
    "completeness_score",
    "confusion_matrix",
    "contingency_matrix",
    "d2_absolute_error_score",
    "d2_pinball_score",
    "d2_tweedie_score",
    "davies_bouldin_score",
    "demographic_parity_difference",
    "disparate_impact_ratio",
    "equal_opportunity_difference",
    "equalized_odds_difference",
    "evaluate",
    "expected_calibration_error",
    "expected_cost_curve",
    "fowlkes_mallows_score",
    "gini_coefficient",
    "group_fairness_report",
    "hit_rate_at_k",
    "homogeneity_score",
    "ks_statistic",
    "lift_table",
    "map_at_k",
    "maximum_calibration_error",
    "mean_reciprocal_rank",
    "multiclass_averages",
    "mutual_info_score",
    "ndcg_at_k",
    "normalized_mutual_info_score",
    "pair_confusion_matrix",
    "precision_at_k",
    "prediction_interval_coverage",
    "predictive_parity_difference",
    "rand_score",
    "recall_at_k",
    "residual_summary",
    "roc_auc",
    "threshold_sweep",
    "top_k_accuracy",
    "v_measure_score",
]
