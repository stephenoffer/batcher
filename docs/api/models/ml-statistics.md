# Statistics, drift, and validation API

This page is the reference for the statistical surface of `batcher.ml`: distribution
statistics and drift comparisons, fairness metrics, resampling for imbalanced labels,
cross-validated scoring and splits, and model interpretation. These are the numbers you
compute *around* a model rather than inside it. Every one is an engine query.

## Statistics and drift

`batcher.ml.stats` covers the statistics that need more than one aggregate, and the
reference-versus-current comparisons a deployed model needs. See the
{doc}`statistics and drift guide </ml/evaluation/statistics-and-drift>`.

```{eval-rst}
.. currentmodule:: batcher.ml.stats

.. autosummary::
   :toctree: generated
   :nosignatures:

   spearman_corr
   entropy
   normalized_entropy
   gini_impurity
   herfindahl_index
   mode_share
   chi_square
   cramers_v
   mutual_information
   anova_f
   theils_u
   eta_squared
   epsilon_squared
   omega_squared
   cohens_f
   correlation_matrix
   covariance_matrix
   partial_correlation
   variance_inflation_factor
   trimmed_mean
   winsorized_mean
   median_abs_deviation
   mean_abs_deviation
   outlier_mask
   population_stability_index
   kl_divergence
   js_divergence
   categorical_drift
   woe_table
   information_value
   drift_report
   TestResult
   t_test_1samp
   t_test_ind
   anova_test
   chi_square_test
   normality_test
   pearson_test
   spearman_test
   proportion_ztest
   binomial_test
   mcnemar_test
   bartlett_test
   levene_test
   mann_whitney_u
   wilcoxon_signed_rank
   kruskal_wallis
   friedman_test
   cliffs_delta
   common_language_effect_size
```

## Fairness

`batcher.ml.metrics` includes the fairness metrics, the grouped comparisons that reveal a model
treating a protected group differently, each a single grouped aggregate.

```{eval-rst}
.. currentmodule:: batcher.ml.metrics

.. autosummary::
   :toctree: generated
   :nosignatures:

   demographic_parity_difference
   disparate_impact_ratio
   equal_opportunity_difference
   equalized_odds_difference
   predictive_parity_difference
   group_fairness_report
   d2_tweedie_score
   d2_absolute_error_score
   d2_pinball_score
```

## Resampling for imbalanced learning

`batcher.ml.sampling` reshapes the class balance as a relational operation: an exact
content-hashed filter or concatenation, never a driver-side shuffle.

```{eval-rst}
.. currentmodule:: batcher.ml.sampling

.. autosummary::
   :toctree: generated
   :nosignatures:

   class_counts
   class_weights
   sample_weights
   undersample
   oversample
   balanced_sample
   stratified_sample
```

## Cross-validated scoring

`batcher.ml.model_selection` ties the fold splitter, a fitted model, and a metric into one
loop, and each fold's data runs through the engine rather than a driver-held array.

```{eval-rst}
.. currentmodule:: batcher.ml.model_selection

.. autosummary::
   :toctree: generated
   :nosignatures:

   cross_val_score
   cross_val_predict
   learning_curve
   validation_curve
```

## Hyperparameter search

The same fold loop, run once per parameter combination. Every combination is scored on the
same folds, so the comparison between two of them is paired rather than confounded with
fold-assignment luck.

```{eval-rst}
.. currentmodule:: batcher.ml.model_selection

.. autosummary::
   :toctree: generated
   :nosignatures:

   param_grid
   param_samples
   grid_search
   random_search
   SearchResult
```

## Cross-validation splits

`batcher.ml.splitting` builds folds as content-hash filters rather than a materialized
shuffle, so a fold is an ordinary row-wise predicate and the assignment is identical
however the data is partitioned.

```{eval-rst}
.. currentmodule:: batcher.ml.splitting

.. autosummary::
   :toctree: generated
   :nosignatures:

   kfold
   stratified_kfold
   stratified_split
   group_kfold
   time_series_split
   fold_column
```

## Model interpretation

`batcher.ml.interpret` explains a model over the whole dataset rather than a driver-sized
sample, because both techniques re-score through the engine.

```{eval-rst}
.. currentmodule:: batcher.ml.interpret

.. autosummary::
   :toctree: generated
   :nosignatures:

   permutation_importance
   partial_dependence
```

## See also

- {doc}`/api/models/ml-models`: the estimators and metrics these statistics are computed around.
- {doc}`/ml/evaluation/statistics-and-drift`: the guide, with the monitoring workflow.
- {doc}`/ml/evaluation/splits-and-resampling`: the guide to the resampling and splitting functions above.
- {doc}`/ml/evaluation/evaluation`: per-segment scoring and the diagnostic tables.
- {doc}`/cookbook/metrics/statistics/index`: 6 runnable recipes for these statistics in the engine.
