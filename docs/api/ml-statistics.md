# Statistics, drift, and validation API

This page is the reference for the statistical surface of `batcher.ml`: distribution
statistics and drift comparisons, fairness metrics, resampling for imbalanced labels,
cross-validated scoring and splits, and model interpretation.

These are the numbers you compute *around* a model rather than inside it, and every one
of them is an engine query.

## Statistics and drift

`batcher.ml.stats` covers the statistics that need more than one aggregate, and the
reference-versus-current comparisons a deployed model needs. See the
[statistics and drift guide](../ml/statistics-and-drift.md).

```{eval-rst}
.. currentmodule:: batcher.ml.stats

.. autofunction:: spearman_corr
.. autofunction:: entropy
.. autofunction:: gini_impurity
.. autofunction:: herfindahl_index
.. autofunction:: mode_share
.. autofunction:: chi_square
.. autofunction:: cramers_v
.. autofunction:: mutual_information
.. autofunction:: anova_f
.. autofunction:: theils_u
.. autofunction:: eta_squared
.. autofunction:: epsilon_squared
.. autofunction:: omega_squared
.. autofunction:: cohens_f
.. autofunction:: correlation_matrix
.. autofunction:: covariance_matrix
.. autofunction:: partial_correlation
.. autofunction:: variance_inflation_factor
.. autofunction:: trimmed_mean
.. autofunction:: winsorized_mean
.. autofunction:: median_abs_deviation
.. autofunction:: outlier_mask
.. autofunction:: population_stability_index
.. autofunction:: kl_divergence
.. autofunction:: js_divergence
.. autofunction:: categorical_drift
.. autofunction:: woe_table
.. autofunction:: information_value
.. autofunction:: drift_report
.. autoclass:: TestResult
.. autofunction:: t_test_1samp
.. autofunction:: t_test_ind
.. autofunction:: anova_test
.. autofunction:: chi_square_test
.. autofunction:: normality_test
.. autofunction:: pearson_test
.. autofunction:: spearman_test
.. autofunction:: proportion_ztest
.. autofunction:: binomial_test
.. autofunction:: mcnemar_test
.. autofunction:: bartlett_test
.. autofunction:: levene_test
.. autofunction:: mann_whitney_u
.. autofunction:: wilcoxon_signed_rank
.. autofunction:: kruskal_wallis
.. autofunction:: friedman_test
.. autofunction:: cliffs_delta
.. autofunction:: common_language_effect_size
```

## Fairness

`batcher.ml.metrics` includes the fairness metrics — grouped comparisons that reveal a model
treating a protected group differently, each a single grouped aggregate.

```{eval-rst}
.. currentmodule:: batcher.ml.metrics

.. autofunction:: demographic_parity_difference
.. autofunction:: disparate_impact_ratio
.. autofunction:: equal_opportunity_difference
.. autofunction:: equalized_odds_difference
.. autofunction:: predictive_parity_difference
.. autofunction:: group_fairness_report
.. autofunction:: d2_tweedie_score
.. autofunction:: d2_absolute_error_score
.. autofunction:: d2_pinball_score
```

## Resampling for imbalanced learning

`batcher.ml.sampling` reshapes the class balance as a relational operation — an exact
content-hashed filter or concatenation, never a driver-side shuffle.

```{eval-rst}
.. currentmodule:: batcher.ml.sampling

.. autofunction:: class_counts
.. autofunction:: class_weights
.. autofunction:: sample_weights
.. autofunction:: undersample
.. autofunction:: oversample
.. autofunction:: balanced_sample
.. autofunction:: stratified_sample
```

## Cross-validated scoring

`batcher.ml.model_selection` ties the fold splitter, a fitted model, and a metric into one
loop — each fold's data runs through the engine rather than a driver-held array.

```{eval-rst}
.. currentmodule:: batcher.ml.model_selection

.. autofunction:: cross_val_score
.. autofunction:: cross_val_predict
.. autofunction:: learning_curve
.. autofunction:: validation_curve
```

## Cross-validation splits

`batcher.ml.splitting` builds folds as content-hash filters rather than a materialized
shuffle, so a fold is an ordinary row-wise predicate and the assignment is identical
however the data is partitioned.

```{eval-rst}
.. currentmodule:: batcher.ml.splitting

.. autofunction:: kfold
.. autofunction:: stratified_kfold
.. autofunction:: group_kfold
.. autofunction:: time_series_split
.. autofunction:: fold_column
```

## Model interpretation

`batcher.ml.interpret` explains a model over the whole dataset rather than a driver-sized
sample, because both techniques re-score through the engine.

```{eval-rst}
.. currentmodule:: batcher.ml.interpret

.. autofunction:: permutation_importance
.. autofunction:: partial_dependence
```

- [Inference](../ml/inference.md): batch prediction and embeddings.
- [Preprocessors](../ml/preprocessors/index.md): fit/transform feature engineering.
- [Multimodal](../ml/multimodal.md): download, decode, tensors, vector search.
- [Serving](../ml/serving.md) and [LLM inference](../ml/llm.md).
- [PyTorch](../ml/pytorch.md) and [streaming](../ml/streaming.md) training loaders.
- [GPU scheduling](../ml/gpu.md): how `num_gpus` and `concurrency` map to actors.
- [Tabular models](../ml/tabular-models.md): scoring XGBoost, LightGBM, and scikit-learn.
- [Evaluation](../ml/evaluation.md): metrics, per-segment scoring, diagnostic tables.
- [Statistics and drift](../ml/statistics-and-drift.md): feature screening and monitoring.

## See also

:::{seealso}
- {doc}`ml-models`: the estimators and metrics these statistics are computed around.
- {doc}`../ml/statistics-and-drift`: the guide, with the monitoring workflow.
- {doc}`../ml/evaluation`: per-segment scoring and the diagnostic tables.
:::
