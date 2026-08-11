# Models and evaluation API

This page is the reference for the model half of `batcher.ml`: the tabular inference
plane, the estimators that fit inside the engine, the metrics that need a global
ordering, and the selection and interpretation helpers.

The single-pass metric *expressions* such as {py:func}`bt.rmse <batcher.rmse>` and {py:func}`bt.f1_score <batcher.f1_score>` live in
{doc}`/api/relational/expressions` instead, because they are ordinary aggregates.

## Tabular models

`batcher.ml.tabular` is the classical-ML inference plane behind
{py:meth}`ds.ml.predict() <batcher.api.dataset.ml.DatasetML.predict>`: it assembles Arrow columns into the dense matrix an
XGBoost, LightGBM, CatBoost, scikit-learn, or ONNX model expects, and wraps the model as a
load-once class UDF. See the {doc}`tabular models guide </ml/inference/tabular-models>`.

```{eval-rst}
.. currentmodule:: batcher.ml.tabular

.. autofunction:: tabular_predictor
.. autofunction:: predicted_column_names
.. autofunction:: feature_matrix
.. autofunction:: prediction_columns
.. autofunction:: resolve_features
.. autofunction:: detect_framework
.. autofunction:: get_adapter
```

## Linear models

`batcher.ml.linear` fits ordinary and ridge regression inside the engine. The normal equations
are built from the feature/target moments, so the whole fit is a single scan and only the small
solve runs on the driver. Both reproduce scikit-learn's coefficients exactly.

```{eval-rst}
.. currentmodule:: batcher.ml.linear

.. autoclass:: LinearRegression
   :members:

.. autoclass:: Ridge
   :members:

.. autoclass:: RidgeClassifier
   :members:

.. autoclass:: RidgeCV
   :members:

.. autoclass:: LogisticRegression
   :members:
```

`batcher.ml.dummy` holds the baseline predictors a real model must beat.

```{eval-rst}
.. currentmodule:: batcher.ml.dummy

.. autoclass:: DummyRegressor
   :members:

.. autoclass:: DummyClassifier
   :members:
```

`batcher.ml.sparse_linear` adds the L1-regularized linear models that select features by zeroing coefficients.

```{eval-rst}
.. currentmodule:: batcher.ml.sparse_linear

.. autoclass:: Lasso
   :members:

.. autoclass:: ElasticNet
   :members:

.. autoclass:: LassoCV
   :members:

.. autoclass:: ElasticNetCV
   :members:
```

`batcher.ml.glm` fits the Tweedie family of generalized linear models by the same one-scan IRLS steps, covering the general form and its Poisson and gamma special cases.

```{eval-rst}
.. currentmodule:: batcher.ml.glm

.. autoclass:: TweedieRegressor
   :members:

.. autoclass:: PoissonRegressor
   :members:

.. autoclass:: GammaRegressor
   :members:

.. autoclass:: HuberRegressor
   :members:
```

`batcher.ml.naive_bayes` adds the probabilistic baseline whose entire fit is one grouped
aggregate.

```{eval-rst}
.. currentmodule:: batcher.ml.naive_bayes

.. autoclass:: GaussianNB
   :members:

.. autoclass:: MultinomialNB
   :members:

.. autoclass:: BernoulliNB
   :members:
```

`batcher.ml.discriminant` adds the Gaussian classifiers: {py:class}`LinearDiscriminantAnalysis <batcher.ml.discriminant.LinearDiscriminantAnalysis>` shares one
covariance across classes (linear boundaries), and {py:class}`QuadraticDiscriminantAnalysis <batcher.ml.discriminant.QuadraticDiscriminantAnalysis>` gives each class
its own (quadratic boundaries).

```{eval-rst}
.. currentmodule:: batcher.ml.discriminant

.. autoclass:: LinearDiscriminantAnalysis
   :members:

.. autoclass:: QuadraticDiscriminantAnalysis
   :members:
```

`batcher.ml.multiclass` extends a two-class estimator to any number of classes.
{py:class}`LogisticRegression <batcher.ml.linear.LogisticRegression>` fits a single weight
vector, so it answers one yes-or-no question and rejects a target with more than two labels.
{py:class}`OneVsRestClassifier <batcher.ml.multiclass.OneVsRestClassifier>` fits one such
model per class and predicts whichever scores highest.

```{eval-rst}
.. currentmodule:: batcher.ml.multiclass

.. autoclass:: OneVsRestClassifier
   :members:
```

`batcher.ml.compose` turns a classifier's raw scores into probabilities you can act on.
A model that separates classes well can still be badly calibrated, meaning a score of 0.9
does not happen nine times in ten, which matters the moment a threshold carries a cost.
{py:class}`CalibratedClassifierCV <batcher.ml.compose.calibration.CalibratedClassifierCV>`
fits the classifier on each cross-validation fold, learns the score-to-probability mapping
on the fold it held out, and averages those mappings. Fitting the calibration on data the
model trained on learns the overconfidence it shows on rows it memorized, which is not the
overconfidence it shows in production.

```{eval-rst}
.. currentmodule:: batcher.ml.compose.calibration

.. autoclass:: CalibratedClassifierCV
   :members:
```

## Evaluation

`batcher.ml.metrics` holds the metrics that need a global ordering or return a table. The
single-pass metric *expressions*, meaning `bt.rmse`, `bt.f1_score`, and the rest, are in the
{doc}`expression reference </api/relational/expressions>` instead, because they are ordinary aggregates. See
the {doc}`evaluation guide </ml/evaluation/evaluation>`.

```{eval-rst}
.. currentmodule:: batcher.ml.metrics

.. autofunction:: evaluate
.. autofunction:: roc_auc
.. autofunction:: average_precision
.. autofunction:: ks_statistic
.. autofunction:: gini_coefficient
.. autofunction:: confusion_matrix
.. autofunction:: threshold_sweep
.. autofunction:: lift_table
.. autofunction:: calibration_curve
.. autofunction:: classification_report
.. autofunction:: multiclass_averages
.. autofunction:: residual_summary
.. autofunction:: prediction_interval_coverage
.. autofunction:: top_k_accuracy
.. autofunction:: expected_calibration_error
.. autofunction:: maximum_calibration_error
.. autofunction:: brier_skill_score

.. autodata:: METRIC_SETS
```

A recommender is scored differently: what matters is the order within *one* query, averaged
over queries. These compute the metric per group and then average, never pooling rows across
groups, which silently rewards a model that ranks one heavy user well and everyone else
badly.

```{eval-rst}
.. autofunction:: precision_at_k
.. autofunction:: recall_at_k
.. autofunction:: hit_rate_at_k
.. autofunction:: mean_reciprocal_rank
.. autofunction:: map_at_k
.. autofunction:: ndcg_at_k
```

Picking a cutoff is the step between a good AUC and a deployed model, and 0.5 is almost never
the right one:

```{eval-rst}
.. autofunction:: best_threshold
.. autofunction:: best_cost_threshold
.. autofunction:: expected_cost_curve
.. autofunction:: compare_models
```

## Outlier detection

`batcher.ml.outliers` finds the rows that do not belong to the same process as the rest, by
the three standard rules (IQR, z-score, MAD), each a per-column bound learned in one aggregate.

```{eval-rst}
.. currentmodule:: batcher.ml.outliers

.. autofunction:: outlier_bounds
.. autofunction:: flag_outliers
.. autofunction:: count_outliers
.. autofunction:: mahalanobis_distance
.. autoclass:: OutlierClipper
   :members:

.. autoclass:: EllipticEnvelope
   :members:
```

## Clustering

`batcher.ml.cluster` holds the unsupervised clusterers. K-means maps each Lloyd iteration onto
one assignment expression and one grouped mean, so the fit is a handful of scans and labeling is
a single streaming pass.

```{eval-rst}
.. currentmodule:: batcher.ml.cluster

.. autoclass:: KMeans
   :members:

.. autoclass:: NearestCentroid
   :members:
```

`batcher.ml.mixture` fits a Gaussian mixture, giving soft clustering and density estimation by EM.

```{eval-rst}
.. currentmodule:: batcher.ml.mixture

.. autoclass:: GaussianMixture
   :members:
```

Score a clustering against a reference labeling with `batcher.ml.metrics`, each computed from one
{py:meth}`group_by <batcher.Dataset.group_by>` contingency table.

```{eval-rst}
.. currentmodule:: batcher.ml.metrics

.. autofunction:: adjusted_rand_score
.. autofunction:: rand_score
.. autofunction:: normalized_mutual_info_score
.. autofunction:: adjusted_mutual_info_score
.. autofunction:: mutual_info_score
.. autofunction:: homogeneity_score
.. autofunction:: completeness_score
.. autofunction:: v_measure_score
.. autofunction:: fowlkes_mallows_score
.. autofunction:: contingency_matrix
.. autofunction:: pair_confusion_matrix
.. autofunction:: calinski_harabasz_score
.. autofunction:: davies_bouldin_score
```

## Pipelines

Preprocessing and a model as one fitted object, so the sequence `predict` replays is by
construction the one `fit` used:

```{eval-rst}
.. currentmodule:: batcher.ml

.. autoclass:: Pipeline
   :members:

.. autoclass:: TransformedTargetRegressor
   :members:

.. autoclass:: MultiOutputRegressor
   :members:

.. autoclass:: MultiOutputClassifier
   :members:
```

## Nearest neighbours

Prediction and imputation by local similarity. All three fold a bounded reference set into
the expression, so scoring is one projection rather than a join:

```{eval-rst}
.. currentmodule:: batcher.ml

.. autoclass:: KNeighborsRegressor
   :members:

.. autoclass:: KNeighborsClassifier
   :members:

.. autoclass:: KNNImputer
   :members:

.. autofunction:: smote
```

## Model persistence

A fitted estimator has to outlive the process that fitted it, or a model trained across a
cluster cannot be moved anywhere. These write it as readable JSON, the same format the
preprocessors use:

```{eval-rst}
.. currentmodule:: batcher.ml.persistence

.. autofunction:: save_model
.. autofunction:: load_model
.. autofunction:: model_to_dict
.. autofunction:: model_from_dict
```

## Ensembling

Combining several models into one prediction. `blend_predictions` is a weighted average and
needs no fit; `StackingEnsemble` fits a meta-model on out-of-fold predictions, so the
meta-model never sees a base model scoring a row it was fitted on.

```{eval-rst}
.. currentmodule:: batcher.ml.ensemble

.. autofunction:: blend_predictions
.. autofunction:: majority_vote
.. autofunction:: out_of_fold_features

.. autoclass:: StackingEnsemble
   :members:
```

## Feature selection

`batcher.ml.selection` decides which columns are worth keeping without fitting a model,
which is both cheaper and less circular than reading a model's own importances.

```{eval-rst}
.. currentmodule:: batcher.ml.selection

.. autofunction:: feature_report
.. autofunction:: feature_profile
.. autofunction:: constant_columns
.. autofunction:: correlated_columns
```

`batcher.ml.feature_scores` ranks each feature against the target with a univariate score,
the filter half of scikit-learn's `SelectKBest`.

```{eval-rst}
.. currentmodule:: batcher.ml.feature_scores

.. autofunction:: f_classif_scores
.. autofunction:: f_regression_scores
.. autofunction:: chi2_scores
.. autofunction:: mutual_info_scores
.. autofunction:: select_k_best
```

`batcher.ml.timeseries` diagnoses serial structure with the autocorrelation function and the standard tests of whether a series or a model's residuals still carry it.

```{eval-rst}
.. currentmodule:: batcher.ml.timeseries

.. autofunction:: autocorrelation
.. autofunction:: autocorrelations
.. autofunction:: partial_autocorrelation
.. autofunction:: partial_autocorrelations
.. autofunction:: ljung_box
.. autofunction:: durbin_watson
.. autofunction:: mean_absolute_scaled_error
```

## See also

- {doc}`/api/models/ml-statistics`: drift, fairness, resampling, and cross-validation.
- {doc}`/ml/inference/tabular-models`: the guide to scoring a fitted model.
- {doc}`/ml/evaluation/evaluation`: the guide to metrics and per-segment scoring.
- {doc}`/cookbook/ml/index`: 16 runnable recipes across the `batcher.ml` surface.
