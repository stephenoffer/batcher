# Models and evaluation API

This page is the reference for the model half of `batcher.ml`: the tabular inference
plane, the estimators that fit inside the engine, the metrics that need a global
ordering, and the selection and interpretation helpers.

The single-pass metric *expressions* such as {py:func}`bt.rmse <batcher.rmse>` and {py:func}`bt.f1_score <batcher.f1_score>` live on
{doc}`/api/models/metrics` instead, because they are ordinary aggregates.

## Tabular models

`batcher.ml.tabular` is the classical-ML inference plane behind
{py:meth}`ds.ml.predict() <batcher.api.dataset.ml.DatasetML.predict>`: it assembles Arrow columns into the dense matrix an
XGBoost, LightGBM, CatBoost, scikit-learn, or ONNX model expects, and wraps the model as a
load-once class UDF. See the {doc}`tabular models guide </ml/inference/tabular-models>`.

```{eval-rst}
.. currentmodule:: batcher.ml.tabular

.. autosummary::
   :toctree: generated
   :nosignatures:

   tabular_predictor
   predicted_column_names
   feature_matrix
   prediction_columns
   resolve_features
   detect_framework
   get_adapter
```

## Linear models

`batcher.ml.linear` fits ordinary and ridge regression inside the engine. The normal equations
are built from the feature/target moments, so the whole fit is a single scan and only the small
solve runs on the driver. Both reproduce scikit-learn's coefficients exactly.

```{eval-rst}
.. currentmodule:: batcher.ml.linear

.. autosummary::
   :toctree: generated
   :nosignatures:

   LinearRegression
   Ridge
   RidgeClassifier
   RidgeCV
   LogisticRegression
```

`batcher.ml.dummy` holds the baseline predictors a real model must beat.

```{eval-rst}
.. currentmodule:: batcher.ml.dummy

.. autosummary::
   :toctree: generated
   :nosignatures:

   DummyRegressor
   DummyClassifier
```

`batcher.ml.sparse_linear` adds the L1-regularized linear models that select features by zeroing coefficients.

```{eval-rst}
.. currentmodule:: batcher.ml.sparse_linear

.. autosummary::
   :toctree: generated
   :nosignatures:

   Lasso
   ElasticNet
   LassoCV
   ElasticNetCV
```

`batcher.ml.glm` fits the Tweedie family of generalized linear models by the same one-scan IRLS steps, covering the general form and its Poisson and gamma special cases.

```{eval-rst}
.. currentmodule:: batcher.ml.glm

.. autosummary::
   :toctree: generated
   :nosignatures:

   TweedieRegressor
   PoissonRegressor
   GammaRegressor
   HuberRegressor
```

`batcher.ml.naive_bayes` adds the probabilistic baseline whose entire fit is one grouped
aggregate.

```{eval-rst}
.. currentmodule:: batcher.ml.naive_bayes

.. autosummary::
   :toctree: generated
   :nosignatures:

   GaussianNB
   MultinomialNB
   BernoulliNB
```

`batcher.ml.discriminant` adds the Gaussian classifiers: {py:class}`LinearDiscriminantAnalysis <batcher.ml.discriminant.LinearDiscriminantAnalysis>` shares one
covariance across classes (linear boundaries), and {py:class}`QuadraticDiscriminantAnalysis <batcher.ml.discriminant.QuadraticDiscriminantAnalysis>` gives each class
its own (quadratic boundaries).

```{eval-rst}
.. currentmodule:: batcher.ml.discriminant

.. autosummary::
   :toctree: generated
   :nosignatures:

   LinearDiscriminantAnalysis
   QuadraticDiscriminantAnalysis
```

`batcher.ml.multiclass` extends a two-class estimator to any number of classes.
{py:class}`LogisticRegression <batcher.ml.linear.LogisticRegression>` fits a single weight
vector, so it answers one yes-or-no question and rejects a target with more than two labels.
{py:class}`OneVsRestClassifier <batcher.ml.multiclass.OneVsRestClassifier>` fits one such
model per class and predicts whichever scores highest.

```{eval-rst}
.. currentmodule:: batcher.ml.multiclass

.. autosummary::
   :toctree: generated
   :nosignatures:

   OneVsRestClassifier
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

.. autosummary::
   :toctree: generated
   :nosignatures:

   CalibratedClassifierCV
```

## Evaluation

`batcher.ml.metrics` holds the metrics that need a global ordering or return a table.
Everything here consumes a whole column at once. See the
{doc}`evaluation guide </ml/evaluation/evaluation>`.

```{eval-rst}
.. currentmodule:: batcher.ml.metrics

.. autosummary::
   :toctree: generated
   :nosignatures:

   evaluate
   roc_auc
   average_precision
   ks_statistic
   gini_coefficient
   confusion_matrix
   threshold_sweep
   lift_table
   calibration_curve
   classification_report
   multiclass_averages
   residual_summary
   prediction_interval_coverage
   top_k_accuracy
   expected_calibration_error
   maximum_calibration_error
   brier_skill_score

.. autodata:: METRIC_SETS
```

A recommender is scored differently. The order within *one* query decides it, averaged over
queries, so these compute the metric per group and then average. Pooling rows across groups
silently rewards a model that ranks one heavy user well and everyone else badly.

```{eval-rst}
.. currentmodule:: batcher.ml.metrics

.. autosummary::
   :toctree: generated
   :nosignatures:

   precision_at_k
   recall_at_k
   hit_rate_at_k
   mean_reciprocal_rank
   map_at_k
   ndcg_at_k
```

Picking a cutoff is the step between a good AUC and a deployed model, and 0.5 is almost never
the right one:

```{eval-rst}
.. currentmodule:: batcher.ml.metrics

.. autosummary::
   :toctree: generated
   :nosignatures:

   best_threshold
   best_cost_threshold
   expected_cost_curve
   compare_models
```

## Outlier detection

`batcher.ml.outliers` finds the rows that do not belong to the same process as the rest, by
the three standard rules (IQR, z-score, MAD), each a per-column bound learned in one aggregate.

```{eval-rst}
.. currentmodule:: batcher.ml.outliers

.. autosummary::
   :toctree: generated
   :nosignatures:

   outlier_bounds
   flag_outliers
   count_outliers
   mahalanobis_distance
   OutlierClipper
   EllipticEnvelope
```

## Clustering

`batcher.ml.cluster` holds the unsupervised clusterers. K-means maps each Lloyd iteration onto
one assignment expression and one grouped mean, so the fit is a handful of scans and labeling is
a single streaming pass.

```{eval-rst}
.. currentmodule:: batcher.ml.cluster

.. autosummary::
   :toctree: generated
   :nosignatures:

   KMeans
   NearestCentroid
```

`batcher.ml.mixture` fits a Gaussian mixture by EM, which gives soft cluster membership and a density estimate at once.

```{eval-rst}
.. currentmodule:: batcher.ml.mixture

.. autosummary::
   :toctree: generated
   :nosignatures:

   GaussianMixture
```

Score a clustering against a reference labeling with `batcher.ml.metrics`, each computed from one
{py:meth}`group_by <batcher.Dataset.group_by>` contingency table.

```{eval-rst}
.. currentmodule:: batcher.ml.metrics

.. autosummary::
   :toctree: generated
   :nosignatures:

   adjusted_rand_score
   rand_score
   normalized_mutual_info_score
   adjusted_mutual_info_score
   mutual_info_score
   homogeneity_score
   completeness_score
   v_measure_score
   fowlkes_mallows_score
   contingency_matrix
   pair_confusion_matrix
   calinski_harabasz_score
   davies_bouldin_score
```

## Pipelines

Preprocessing and a model as one fitted object, so the sequence `predict` replays is by
construction the one `fit` used:

```{eval-rst}
.. currentmodule:: batcher.ml

.. autosummary::
   :toctree: generated
   :nosignatures:

   Pipeline
   TransformedTargetRegressor
   MultiOutputRegressor
   MultiOutputClassifier
```

## Nearest neighbours

Prediction and imputation by local similarity. All three fold a bounded reference set into
the expression, so scoring is one projection rather than a join:

```{eval-rst}
.. currentmodule:: batcher.ml

.. autosummary::
   :toctree: generated
   :nosignatures:

   KNeighborsRegressor
   KNeighborsClassifier
   KNNImputer
   smote
```

## Model persistence

A fitted estimator has to outlive the process that fitted it, or a model trained across a
cluster cannot be moved anywhere. These write it as readable JSON, the same format the
preprocessors use:

```{eval-rst}
.. currentmodule:: batcher.ml.persistence

.. autosummary::
   :toctree: generated
   :nosignatures:

   save_model
   load_model
   model_to_dict
   model_from_dict
```

## Ensembling

Combining several models into one prediction. `blend_predictions` is a weighted average and
needs no fit. `StackingEnsemble` fits a meta-model on out-of-fold predictions, so the
meta-model never sees a base model scoring a row it was fitted on.

```{eval-rst}
.. currentmodule:: batcher.ml.ensemble

.. autosummary::
   :toctree: generated
   :nosignatures:

   blend_predictions
   majority_vote
   out_of_fold_features
   StackingEnsemble
```

## Feature selection

`batcher.ml.selection` decides which columns are worth keeping without fitting a model,
which is both cheaper and less circular than reading a model's own importances.

```{eval-rst}
.. currentmodule:: batcher.ml.selection

.. autosummary::
   :toctree: generated
   :nosignatures:

   feature_report
   feature_profile
   constant_columns
   correlated_columns
```

`batcher.ml.feature_scores` ranks each feature against the target with a univariate score,
the filter half of scikit-learn's `SelectKBest`.

```{eval-rst}
.. currentmodule:: batcher.ml.feature_scores

.. autosummary::
   :toctree: generated
   :nosignatures:

   f_classif_scores
   f_regression_scores
   chi2_scores
   mutual_info_scores
   select_k_best
```

`batcher.ml.timeseries` diagnoses serial structure with the autocorrelation function and the standard tests of whether a series or a model's residuals still carry it.

```{eval-rst}
.. currentmodule:: batcher.ml.timeseries

.. autosummary::
   :toctree: generated
   :nosignatures:

   autocorrelation
   autocorrelations
   partial_autocorrelation
   partial_autocorrelations
   ljung_box
   durbin_watson
   mean_absolute_scaled_error
```

## See also

- {doc}`/api/models/ml-statistics`: drift, fairness, resampling, and cross-validation.
- {doc}`/ml/inference/tabular-models`: the guide to scoring a fitted model.
- {doc}`/ml/evaluation/evaluation`: the guide to metrics and per-segment scoring.
- {doc}`/cookbook/ml/index`: 27 runnable recipes across the `batcher.ml` surface.
