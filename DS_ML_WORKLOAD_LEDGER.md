# Data-Science / Traditional-ML Workload Ledger

Working ledger for the "make Batcher a first-class engine for data science and classical
(non-LLM) machine learning" effort: gradient-boosted-tree batch inference, sklearn-style
preprocessing, statistical expressions, evaluation metrics, drift monitoring, and the
cross-validation / feature-engineering surface a DS workflow actually needs.

Companion to `AI_WORKLOAD_LEDGER.md` (LLM / GPU / multimodal). This one is the tabular half.

## Design rules this effort holds to

- **No per-row Python.** Every metric, statistic, and drift measure lowers to relational
  aggregates / window functions, so it is mergeable, distributed, and spillable for free.
  The only Python that touches a row is inside a `map_batches` UDF that a *model* needs.
- **Preprocessors follow the existing contract** (`ml/preprocessors/base.py`): `fit` runs
  a bounded number of mergeable aggregates, `transform` returns lazy `Expr` projections.
- **Model inference is a load-once class UDF**, so the distributed warm pool reuses it.
- **Every metric is checked against an oracle** — scikit-learn for the metrics, SciPy for
  the statistics — at exact equality wherever the two compute the same closed form.

## Status key

`[x]` done and verified · `[~]` in progress · `[ ]` not started

---

## Cluster A — Gradient-boosted-tree and sklearn batch inference

The headline gap: `ds.ml.infer` only knew HuggingFace model ids. There was no way to score
an XGBoost / LightGBM / CatBoost / scikit-learn / ONNX model over a Dataset at all.

- [x] A.1 `ml/tabular/` package — the tabular-model inference plane.
- [x] A.2 `feature_matrix` — assemble N columns into one dense row-major array per batch,
  in a fixed feature order.
- [x] A.3 Nulls become NaN (the boosters' own missing convention), overridable per model.
- [x] A.4 `XGBoostAdapter` — `Booster` and sklearn wrapper, DMatrix, margin, leaf indices,
  SHAP contributions, `iteration_range`.
- [x] A.5 `LightGBMAdapter` — `Booster` and sklearn wrapper, raw score, leaf, contributions,
  `num_iteration`.
- [x] A.6 `CatBoostAdapter` — `CatBoost*` models, raw formula values, leaf indexes, SHAP.
- [x] A.7 `SklearnAdapter` — any fitted estimator or `Pipeline`, incl. duck-typed ones.
- [x] A.8 `ONNXAdapter` — ONNX Runtime session with CUDA-then-CPU provider selection.
- [x] A.9 Model loading from a path or cloud URI, fetched once per worker.
- [x] A.10 `ds.ml.predict(...)` — the Dataset entry point, dispatching on object or path.
- [x] A.11 Multi-class output as wide columns or one list column (`as_list=`).
- [x] A.12 Per-worker thread pinning so co-located actors don't oversubscribe the host.
- [x] A.13 Memoized predictor classes, so the distributed warm-pool key stays stable.
- [x] A.14 Output width derived from the model at plan time; a saved model is opened once
  (cached per path) rather than assumed single-output.
- [x] A.15 Feature-order guard: a count mismatch or a permutation of the model's own
  feature names raises at plan time instead of silently changing every prediction.
- [x] A.16 Per-framework default matrix precision — float32 for the boosters (their own
  internal precision), float64 for scikit-learn, whose output otherwise shifts.
- [x] A.17 An all-null feature column in a batch is treated as missing, not as a type error.
- [x] A.18 A zero-row batch keeps the output schema instead of calling the model.
- [x] A.19 `predict_proba` on a bare `Booster` is an actionable error, not a silent
  probability.
- [x] A.20 `xgboost` / `lightgbm` / `catboost` / `onnx` / `sklearn` extras plus a `tabular`
  bundle; the boosters added to `dev` so CI actually exercises those paths.

## Cluster B — Evaluation metrics as relational aggregates

- [x] B.1 `plan/functions/metrics/` — metrics as `Expr`, so they compose with `group_by`
  and a per-segment report costs one pass.
- [x] B.2-B.16 Regression: `mse`, `rmse`, `mae`, `medae`, `max_error`, `mean_bias`, `mape`,
  `smape`, `wape`, `r2`, `explained_variance`, `msle`, `rmsle`, `huber_loss`,
  `pinball_loss`.
- [x] B.17-B.34 Classification: `true_positives`/`false_positives`/`false_negatives`/
  `true_negatives`, `accuracy`, `precision`, `recall`, `specificity`,
  `false_positive_rate`, `negative_predictive_value`, `f1_score`, `fbeta_score`,
  `balanced_accuracy`, `matthews_corrcoef`, `cohen_kappa`, `prevalence`.
- [x] B.35-B.36 Probabilistic: `log_loss` (clipped as scikit-learn does), `brier_score`.
- [x] B.37 `ml/metrics/ranked.py` — `roc_auc` via the rank identity (exact under ties, no
  threshold sweep), `average_precision`, `ks_statistic`, `gini_coefficient`.
- [x] B.38 Every rank metric takes `by=`, giving per-segment AUC in one partitioned window.
- [x] B.39-B.42 Diagnostic tables as lazy Datasets: `confusion_matrix` (long form,
  multi-class), `threshold_sweep`, `lift_table`, `calibration_curve`.
- [x] B.43 `ds.ml.evaluate(...)` / `ml.metrics.evaluate` — a task's whole metric set, the
  aggregates in one pass, hard predictions derived from a score at a threshold.
- [x] B.44 `METRIC_SETS` — the default metric set per task, named rather than assembled.
- [x] B.45 Metric parity: 18 metrics checked against `sklearn.metrics` at 1e-12.
- [x] B.46-B.50 Ranking metrics for recommenders: `precision_at_k`, `recall_at_k`,
  `hit_rate_at_k`, `mean_reciprocal_rank`, `ndcg_at_k` — each computed *within* a query and
  then averaged over queries, never pooled across them.
- [x] B.51 `classification_report` — per-class precision, recall, F1, and support, with
  every class's counts in **one** aggregate rather than one scan per class.
- [x] B.52 `multiclass_averages` — macro and support-weighted averages, exact against
  scikit-learn, so a model that never predicts the minority class cannot pass review on a
  weighted number alone.
- [x] B.53 `task="multiclass"` in `evaluate` now reports the full average set, not just
  accuracy.
- [x] B.54 `residual_summary` — the regression residual (bias, spread, quantiles) grouped by
  segment, surfacing the cohort a global RMSE averages away.
- [x] B.55 `prediction_interval_coverage` — whether a quantile model's "90% interval" covers
  ≈90%, the one claim the model's own output cannot verify.
- [x] B.56 `top_k_accuracy` — the true label among the top `k` guesses, exact against
  scikit-learn.
- [x] B.57-B.66 Diagnostic-test metric vocabulary (all single-pass, over the four confusion
  cells): `jaccard_score`, `false_discovery_rate`, `false_omission_rate`,
  `positive_likelihood_ratio`, `negative_likelihood_ratio`, `diagnostic_odds_ratio`,
  `informedness` (Youden's J), `markedness`, `fowlkes_mallows_index`, `prevalence_threshold`
  — pinned against scikit-learn and against the identities (√(informedness·markedness)=MCC,
  FDR=1−precision, DOR=PLR/NLR).

## Cluster AP — Rank-based non-parametric tests

- [x] AP.6 `friedman_test` — the non-parametric repeated-measures ANOVA, ranking treatments
  *within* each matched block to remove block-to-block variation. Matches SciPy's
  `friedmanchisquare` with and without ties.
- [x] AP.5 `wilcoxon_signed_rank` — the paired distribution-free test (the paired-t replacement),
  ranking the magnitudes of the per-row differences. Matches SciPy's `wilcoxon` (approx mode)
  with and without ties.
- [x] AP.3-AP.4 `cliffs_delta` and `common_language_effect_size` — the non-parametric effect
  sizes to report beside a Mann-Whitney result (the probability one group exceeds the other),
  built on the shared U machinery and verified against the pairwise definition.
- [x] AP.1-AP.2 `mann_whitney_u` and `kruskal_wallis` in `ml/stats/nonparametric.py`: the
  distribution-free alternatives to the two-sample t-test and one-way ANOVA, computed from pooled
  average ranks (ties averaged) with the tie correction as one extra aggregate
  (`sum(tie_size^2 - 1)` = the textbook `sum(t^3 - t)`). Both match SciPy's asymptotic
  `mannwhitneyu` and `kruskal` exactly on statistic and p-value.

## Cluster AO — Variance-homogeneity tests

- [x] AO.1-AO.2 `bartlett_test` and `levene_test` in `ml/stats/homogeneity.py`: the equal-variance
  tests a t-test and ANOVA quietly assume — Bartlett's for normal groups, Levene's median-centered
  form for robustness. Both reduce to per-group aggregates and match SciPy's `bartlett`/`levene`
  exactly on statistic and p-value.

## Cluster AN — Variance inflation factor

- [x] AN.1 `variance_inflation_factor` in `ml/stats/multivariate.py`: the per-feature
  multicollinearity diagnostic (``1 / (1 - R^2_j)``), read off the diagonal of the inverted
  correlation matrix in one scan. Matches the independent regress-each-on-the-rest definition.

## Cluster AM — Margin losses

- [x] AM.1-AM.2 `hinge_loss` and `squared_hinge_loss` metric expressions (`ml/metrics` +
  `plan/functions/metrics/margin.py`): the support-vector-machine objectives, scoring a raw
  decision function rather than a probability. `hinge_loss` matches scikit-learn; the squared
  variant matches its closed form. Surfaced through the `bt.` metrics facade.

## Cluster AQ — Tweedie GLM (unifying Poisson and Gamma)

- [x] AQ.1 `TweedieRegressor` in `ml/glm.py`: the general log-link Tweedie GLM (power in [1,2]),
  fitting the compound Poisson-gamma distribution of a "mostly zero, some positive" target such as
  an insurance pure premium. Matches scikit-learn across powers and penalties. Refactored
  `PoissonRegressor` (power 1) and `GammaRegressor` (power 2) into thin subclasses of it —
  eliminating the duplicated IRLS and proving power-1==Poisson, power-2==Gamma exactly.

## Cluster AL — Poisson regression (count-data GLM)

- [x] AL.2 `GammaRegressor` in `ml/glm.py`: the log-link GLM for a positive, right-skewed
  continuous target (claim sizes, durations, spend), fitted by Fisher-scoring IRLS and matching
  scikit-learn's `GammaRegressor` across penalty strengths.
- [x] AL.1 `PoissonRegressor` in `ml/glm.py`: a log-link generalized linear model for count
  targets, fitted by the same one-scan IRLS Newton steps as logistic regression, with an L2
  penalty matching scikit-learn's convention. Matches scikit-learn's coefficients and predictions
  across penalty strengths. Split the IRLS GLMs into `ml/glm.py` to keep `ml/linear.py` within the
  size limit. Added to the `batcher.ml` public facade.

## Cluster AK — Nearest-centroid and quadratic discriminant classifiers

- [x] AK.1 `NearestCentroid` in `ml/cluster.py`: the supervised counterpart to KMeans — one
  centroid per class, label by the nearest. Matches scikit-learn exactly.
- [x] AK.3 `LinearDiscriminantAnalysis` in `ml/discriminant.py`: the shared-covariance Gaussian
  classifier (linear boundaries, one pooled covariance from all rows), matching scikit-learn for
  both its solvers in and out of sample.
- [x] AK.2 `QuadraticDiscriminantAnalysis` in `ml/discriminant.py`: the Gaussian classifier that
  models each class's *full* covariance (quadratic boundaries), separating classes that differ in
  spread or orientation, which a diagonal-covariance model misses. Fit is per-class mean and
  covariance aggregates; prediction is a quadratic-form argmax. Matches scikit-learn exactly in and
  out of sample. Both added to the `batcher.ml` public facade.

## Cluster AJ — Naive Bayes variants

- [x] AJ.1-AJ.2 `MultinomialNB` and `BernoulliNB` in `ml/naive_bayes.py`: the count-feature and
  binary-feature naive-Bayes classifiers (the text-classification workhorses), fitted from grouped
  feature sums with Laplace smoothing and classifying by a closed-form argmax. Both match
  scikit-learn exactly in and out of sample. Refactored the shared argmax into one helper reused
  by all three NB classes. Added to the `batcher.ml` facade.

## Cluster AY — Internal clustering-quality scores

- [x] AY.1-AY.2 `calinski_harabasz_score` (variance ratio) and `davies_bouldin_score` (average
  worst-case cluster overlap) in `ml/metrics/cluster_quality.py`: score a clustering on its own
  geometry with *no* reference labeling — what an elbow search over the cluster count optimizes.
  Built from per-cluster centroids and dispersions (no pairwise blow-up). Both match scikit-learn
  exactly on structured and random labelings.

## Cluster AI — Clustering-quality metrics

- [x] AI.10-AI.11 Added `contingency_matrix` (the labeled co-occurrence table of two labelings)
  and `pair_confusion_matrix` (the four pair-counting buckets the Rand-family scores are built
  from), both matching scikit-learn. Extracted the shared contingency/entropy math into
  `_cluster_shared.py` to keep `clustering.py` within the size limit.
- [x] AI.9 Added `adjusted_mutual_info_score` — the chance-corrected mutual information (Vinh's
  expected-MI adjustment), the information-theoretic counterpart of `adjusted_rand_score`, matching
  scikit-learn exactly.
- [x] AI.7-AI.8 Added `rand_score` (the raw agreement fraction) and `mutual_info_score` (the
  unnormalized mutual information in nats), both matching scikit-learn.
- [x] AI.1-AI.6 `ml/metrics/clustering.py`: `adjusted_rand_score`, `normalized_mutual_info_score`,
  `homogeneity_score`, `completeness_score`, `v_measure_score`, and `fowlkes_mallows_score` — the
  standard measures of a clustering's agreement with a reference labeling, each a closed-form
  function of one `group_by` contingency table. All six match scikit-learn exactly (NMI uses
  sklearn's default arithmetic normalization).

## Cluster AH — Gaussian naive Bayes

- [x] AH.1 `GaussianNB` in `ml/naive_bayes.py`: a probabilistic classifier whose entire fit — a
  per-class prior, mean, and variance — is a single `group_by(target)` aggregate, and whose
  prediction is a closed-form log-likelihood argmax expression. Matches scikit-learn's `GaussianNB`
  predictions exactly in and out of sample (including its `var_smoothing`). Added to the
  `batcher.ml` public facade.

## Cluster AX — Sparse linear models (lasso and elastic net)

- [x] AX.1-AX.2 `ElasticNet` and `Lasso` in `ml/sparse_linear.py`: L1/elastic-net regularized
  regression that drives irrelevant coefficients to exactly zero (feature selection as it trains).
  Coordinate descent runs on the driver over the centered Gram matrix and feature-target
  covariances (one scan); the strictly convex objective has a unique minimizer, so the
  coefficients, intercept, and sparsity pattern match scikit-learn exactly across alpha/l1_ratio.
  Added to the `batcher.ml` facade.

## Cluster AU — Ridge classification

- [x] AU.1 `RidgeClassifier` in `ml/linear.py`: classification cast as one-vs-rest ridge
  regression (a +1/-1 target per class, argmax of the scores) — a closed-form, single-scan-per-class
  fit, stable under collinear features. Matches scikit-learn's `RidgeClassifier` exactly across
  penalties in and out of sample. Added to the `batcher.ml` facade.

## Cluster AG — Native logistic regression

- [x] AG.1 `LogisticRegression` in `ml/linear.py`: a binary GLM classifier fitted by iteratively
  reweighted least squares — each Newton step's gradient and Hessian are per-row-product
  aggregates (one scan), the small solve runs on the driver, and `predict_proba`/`predict` are
  single-pass expressions. Converges to scikit-learn's unpenalized coefficients, intercept, and
  probabilities. Added to the `batcher.ml` public facade.

## Cluster AF — Native linear models

- [x] AF.1-AF.2 `LinearRegression` and `Ridge` in `ml/linear.py`: ordinary and L2-regularized
  least squares fitted *inside the engine* — the normal equations are built from the feature and
  target moments (one scan), only the small solve runs on the driver, and prediction is a
  linear-combination expression. Both reproduce scikit-learn's coefficients, intercept, and
  predictions exactly across penalty strengths. Added to the `batcher.ml` public facade.

## Cluster AE — Multivariate outliers and the hyperparameter tuning curve

- [x] AE.1 `mahalanobis_distance` in `ml/outliers.py`: the multivariate outlier score for a row
  that looks ordinary on each column but is far from the joint center, using the learned mean and
  inverse covariance. Matches `scipy.spatial.distance.mahalanobis` exactly. Added to the
  `batcher.ml` facade.
- [x] AE.2 `validation_curve` in `ml/model_selection.py`: the cross-validated score as a function
  of one hyperparameter (the bias-variance tuning curve), the `learning_curve` counterpart that
  varies capacity instead of data size.

## Cluster AD — Imbalanced g-mean and scale-free forecast error

- [x] AD.1 `geometric_mean_score` metric expression (`sqrt(recall * specificity)`) — the
  imbalanced-classification score high only when *both* classes are recalled, surfaced through
  the `bt.` metrics facade.
- [x] AD.2 `mean_absolute_scaled_error` in `ml/timeseries.py`: the scale-free forecasting metric
  (model MAE over the seasonal-naive MAE), honoring the time order. Both verified against numpy
  closed forms.

## Cluster AV — Gaussian mixture models

- [x] AV.1 `GaussianMixture` in `ml/mixture.py`: soft clustering and density estimation by
  expectation-maximization with full covariances. The E-step is a per-row responsibility
  expression (numerically-stable log-sum-exp via horizontal max/sum), the M-step a set of weighted
  aggregates; only the per-component matrix inverse runs on the driver. Exposes `predict`,
  `predict_proba`, and `score_samples` (the per-row log-likelihood anomaly score). Verified by
  property: the log-likelihood increases every iteration, the clustering matches truth and
  scikit-learn up to permutation (ARI ~0.99), and an outlier scores far below every inlier. Added
  to the `batcher.ml` facade.

## Cluster AC — K-means clustering

- [x] AC.1 `KMeans` clusterer in `ml/cluster.py`: Lloyd's algorithm mapped onto the engine —
  each iteration is one nearest-centroid assignment expression plus one grouped mean, so the fit
  is a handful of scans and labeling is a single streaming pass. Exposes `centroids_`,
  `inertia_`, `n_iter_`, and `fit`/`predict`/`fit_predict`. On separable blobs it matches both
  ground truth and scikit-learn up to label permutation (adjusted Rand index 1.0) and its inertia
  matches scikit-learn's. Reproducible from a content-hash seed. Added to the `batcher.ml` facade.

## Cluster AR — Truncated SVD

- [x] AR.1 `TruncatedSVD` in `ml/preprocessors/derived/decomposition.py`: dimensionality reduction
  without centering (the reducer for non-negative or sparse feature blocks where centering would
  destroy structure), projecting onto the top singular vectors of the data's second-moment matrix.
  Reproduces scikit-learn's projection and `explained_variance_ratio_`. Added to the `batcher.ml`
  facade.

## Cluster AB — PCA dimensionality reduction

- [x] AB.1 `PCA` preprocessor in `ml/preprocessors/decomposition.py`: projects a block of
  correlated columns onto their top principal components, replacing them with uncorrelated
  `pc1..pck` columns ordered by explained variance. The fit is a single scan (mean + covariance
  are aggregates, only the small eigendecomposition runs on the driver); the transform is a set
  of linear-projection expressions. Reproduces scikit-learn's projection (up to per-component
  sign) and `explained_variance_ratio_` exactly. Added to the `batcher.ml` public facade.

## Cluster AA — Multivariate association

- [x] AA.1-AA.3 `ml/stats/multivariate.py`: `correlation_matrix` and `covariance_matrix` return
  the whole pairwise structure of a feature set as a labeled square `Dataset` from one scan, and
  `partial_correlation` removes a confounder (the correlation of two features after holding a
  third, or several, fixed). Matrices match numpy `corrcoef`/`cov`; the partial correlation
  matches an independent residual-regression computation for one and several controls.

## Cluster Z — Mean average precision (the recommender ranking metric)

- [x] Z.1 `map_at_k` in `ml/metrics/ranking.py`: rank-aware mean average precision, the standard
  single number for a ranked-retrieval or recommender system. Unlike `precision_at_k` it rewards
  ranking relevant items *high*, computed with windowed cumulative precision and matched against
  a position-by-position numpy reference.

## Cluster Y — Proportional stratified subsampling

- [x] Y.1 `stratified_sample` in `ml/sampling.py`: keeps the same fraction of every stratum
  (preserving class balance) rather than equalizing them like the other samplers. Exact
  per-stratum count cut over a content hash, reproducible and partition-independent. Added to
  the `batcher.ml` public facade.

## Cluster AT — Binary categorical encoding

- [x] AT.1 `BinaryEncoder` in `ml/preprocessors/encoders/binary.py`: compact base-2 categorical
  encoding — a category's integer code written across ceil(log2(n+1)) bit columns, so 100
  categories cost 7 columns rather than 100 one-hot, with no collisions. `transform` lowers to a
  `when` chain plus cheap bitwise shifts. Added to the `batcher.ml` facade.

## Cluster W — Box-Cox power transform

- [x] W.1 `BoxCoxTransformer` (and the `box_cox` expression) in `ml/preprocessors/power.py`:
  the maximum-likelihood Box-Cox normalizer, the strictly-positive counterpart to the
  Yeo-Johnson `PowerTransformer`, fit in a single pass (every candidate lambda's profile
  likelihood is one aggregate). The fitted lambda lands within grid resolution of scipy's
  `boxcox` and the transform reproduces scipy exactly at that lambda; a non-positive column
  raises rather than emitting NaNs. Added to the `batcher.ml` public facade.

## Cluster V — D-squared regression scores

- [x] V.1-V.2 `d2_absolute_error_score` and `d2_pinball_score` in `ml/metrics/regression.py`:
  R²-style deviance-explained scores on the L1 and quantile scales, each measuring a model
  against the optimal constant baseline (median for absolute error, alpha-quantile for pinball).
  Both reproduce scikit-learn exactly across quantiles.

## Cluster AW — Exact binomial test

- [x] AW.1 `binomial_test` in `ml/stats/hypothesis.py`: the exact small-sample counterpart of
  `proportion_ztest`, summing the binomial probabilities of every outcome no more likely than the
  observed one. Matches SciPy's `binomtest` two-sided p-value exactly.

## Cluster U — Correlation-significance, proportion, and paired-classifier tests

- [x] U.1-U.4 Four more tests in `ml/stats/hypothesis.py`, on a new `normal` survival function
  (`math.erfc`, exact): `pearson_test` and `spearman_test` (a p-value on a linear or monotone
  correlation, matching scipy's `pearsonr`/`spearmanr`), `proportion_ztest` (a success rate
  against a target), and `mcnemar_test` (the paired test comparing two classifiers' error rates
  on the same rows — the right tool for "does model B beat model A"). Fixed the now-false docs
  note that claimed Batcher returns no p-values.

## Cluster AS — Partial autocorrelation

- [x] AS.1-AS.2 `partial_autocorrelation` and `partial_autocorrelations` in `ml/timeseries.py`:
  the Yule-Walker PACF via the Durbin-Levinson recursion over the sample autocorrelations — the
  tool for choosing an autoregressive order, since the PACF cuts off at the true AR order where
  the ACF only decays. Verified against an independent Yule-Walker Toeplitz solve.

## Cluster T — Time-series diagnostics

- [x] T.1-T.4 `ml/timeseries.py`: `autocorrelation` (Box-Jenkins lag-k r_k), `autocorrelations`
  (the sample ACF), `ljung_box` (portmanteau white-noise test, reusing the `chi2` survival
  function and returning a `TestResult`), and `durbin_watson` (residual serial-correlation
  diagnostic). Each orders the column by a time key, lags it over a window, and reduces the
  overlap to one aggregate. The ordered-window semantics are proven with a shuffled input, and
  every statistic matches an independent numpy reference (statsmodels is not a dependency).

## Cluster AZ — Bias-corrected ANOVA effect sizes

- [x] AZ.1-AZ.2 `omega_squared` (the least-biased variance-explained estimate, for generalizing
  beyond the sample) and `cohens_f` (the effect-size scale a power analysis is specified on) in
  `ml/stats/association.py`, both from the `anova_f` statistic and the degrees of freedom. Verified
  against their sum-of-squares definitions.

## Cluster S — Directional association and ANOVA effect sizes

- [x] S.1-S.3 `theils_u` (asymmetric uncertainty coefficient, the *directional* categorical
  association `cramers_v` cannot express), `eta_squared`, and `epsilon_squared` (the bounded,
  bias-corrected effect sizes that the unbounded `anova_f` lacks) added to
  `ml/stats/association.py`. Verified against closed-form numpy references.

## Cluster R — Univariate feature scoring (the SelectKBest filter)

- [x] R.1-R.5 `ml/feature_scores.py`: `f_classif_scores` (ANOVA F vs a categorical target),
  `f_regression_scores` (regression F vs a continuous target), `chi2_scores`,
  `mutual_info_scores`, and `select_k_best`. Each score is one existing mergeable statistic
  looped over features, so scoring a wide table is one one-pass reduction per column, not a
  materialized correlation matrix. `f_classif_scores` and `f_regression_scores` reproduce
  scikit-learn's `f_classif` / `f_regression` exactly.

## Cluster BB — Closing error, rate, and dispersion measures (reaching 300)

- [x] BB.1-BB.2 `mean_percentage_error` (the signed forecast bias) and `normalized_rmse` (the
  scale-free RMSE) metric expressions.
- [x] BB.3 `false_negative_rate` (`1 - recall`, the miss rate) metric expression.
- [x] BB.4 `mean_abs_deviation` (mean absolute deviation from the mean) in `ml/stats/robust`.
- [x] BB.5 `normalized_entropy` (Shannon entropy scaled to [0,1], comparable across cardinalities)
  in `ml/stats/descriptive`. All five verified against numpy/scipy references. **This batch crosses
  300 distinct new public API names.**

## Cluster BA — Geometric spread and baseline predictors

- [x] BA.1 `geometric_std` — the multiplicative standard deviation (`exp(std(ln x))`) for a
  log-normal column, matching `scipy.stats.gstd`. Surfaced through the `bt.` facade.
- [x] BA.2-BA.3 `DummyRegressor` (mean/median) and `DummyClassifier` (majority class) in
  `ml/dummy.py` — the baseline every real model must beat, matching scikit-learn's `Dummy*`. Added
  to the `batcher.ml` facade.

## Cluster Q — Dispersion ratios, fixed-edge binning, and hypothesis tests

- [x] Q.1-Q.4 Dispersion-ratio expressions in `plan/functions/analysis/moments.py`:
  `index_of_dispersion` (Fano factor), `signal_to_noise`, `studentized_range`, `relative_range`
  — spread expressed relative to level so it is unitless and comparable across columns, each a
  single mergeable aggregate over the moment primitives. Verified against numpy closed forms.
- [x] Q.5 `bt.cut` — fixed-edge binning as a pure `when`/`then` expression (no `fit`), folded
  into `plan/functions/conditional.py` beside `iff`/`nanvl` rather than a 13th functions file.
  Integer index or per-bucket labels, left- or right-open. Matches `numpy.digitize`.
- [x] Q.6-Q.11 Hypothesis tests in `ml/stats/hypothesis.py` returning a `TestResult`
  (statistic, df, p-value): `t_test_1samp`, `t_test_ind` (Welch), `anova_test`,
  `chi_square_test`, `normality_test` (Jarque-Bera). The tail probabilities come from a
  dependency-free `_special.py` (regularized incomplete beta/gamma; Student's t, F, chi-squared
  survival functions), each matched to SciPy at 1e-10 and each test matched to SciPy's own.

## Cluster P — Outlier detection and agreement metrics

- [x] P.1-P.4 `ml/outliers.py`: `outlier_bounds`, `flag_outliers`, `count_outliers`, and the
  `OutlierClipper` preprocessor — three rules (IQR/z-score/MAD), each a per-column bound learned
  in one aggregate, the clipper applying training bounds to serving data.
- [x] P.5-P.7 Agreement/efficiency metrics as expressions (agreement, not just correlation):
  `concordance_correlation` (Lin's CCC), `nash_sutcliffe_efficiency`, `kling_gupta_efficiency`
  — each verified against a reference implementation. Fixed a population-vs-sample moment mix
  in the first CCC draft (gave 2/3 instead of 1 for identical series).

## Cluster O — Cross-validation execution, weighted stats, and more encoders

- [x] O.1-O.3 `ml/model_selection.py`: `cross_val_score` (per-fold scores),
  `cross_val_predict` (out-of-fold predictions covering every row once), `learning_curve`
  (score vs training-set size) — the fold splitter + a fit callable + a metric, tied into
  one loop, verified with a real scikit-learn model.
- [x] O.4-O.8 Weighted statistics as single-pass aggregates: `weighted_var`, `weighted_std`,
  `weighted_covariance`, `weighted_correlation` (and the existing `weighted_mean`), exact
  against `numpy.average` and a frequency-weighted `numpy.cov`.
- [x] O.9-O.11 `RankTransformer` (exact percentile rank), `LabelBinarizer` (one-vs-rest on a
  label), `MultiLabelBinarizer` (a list column to an indicator matrix).
- [x] O.12 `hamming_loss` — the multi-label error rate as an aggregate expression.

## Cluster N — Fairness, count models, and imbalanced learning

- [x] N.1-N.6 Fairness metrics (grouped single-pass): `demographic_parity_difference`,
  `disparate_impact_ratio` (the 80% rule), `equal_opportunity_difference`,
  `equalized_odds_difference`, `predictive_parity_difference`, `group_fairness_report`.
- [x] N.7-N.9 Count/rate deviance losses as expressions: `poisson_deviance`,
  `gamma_deviance`, `tweedie_deviance` (the whole family by `power`), exact against
  scikit-learn's `mean_*_deviance`.
- [x] N.10 `d2_tweedie_score` — deviance-explained R² for a count model, exact vs sklearn.
- [x] N.11-N.16 Imbalanced-learning resampling as *exact* relational operations:
  `class_counts`, `undersample`, `oversample`, `balanced_sample` (to the median),
  `class_weights`, `sample_weights` — content-hash ranked, so exactly balanced and
  reproducible, never a driver-side shuffle.

## Cluster L — Model interpretation at scale

Explaining a model is normally done on a driver-sized sample; both of these re-score through
the engine, so the explanation runs over the same data the model scores.

- [x] L.1 `permutation_importance` — model-agnostic feature importance by shuffling each
  column and measuring the metric drop; verified to rank a coefficient-3 feature above a
  coefficient-2 above a near-useless one, with the useless one at ≈0.
- [x] L.2 `partial_dependence` — the average prediction as one feature sweeps a grid,
  averaged over the real joint distribution of the others; verified to trace a known slope.

## Cluster M — Feature construction and text

- [x] M.1 `Binarizer` · `InteractionFeatures` · `RatioFeatures` (zero-denominator → null) ·
  `ColumnSelector` / `ColumnDropper` · `VarianceThreshold`.
- [x] M.2 `GroupStatEncoder` — per-group statistics joined onto each row (the single most
  productive tabular feature family), learned on train.
- [x] M.3 `GroupImputer` — group-mean imputation with a global-mean fallback.
- [x] M.4 `TextStatFeaturizer` — cheap interpretable text signals (length, word count,
  character mix) as pure string expressions, no model.
- [x] M.5 `WOEEncoder` — weight-of-evidence encoding, the additive-in-log-odds transform a
  regulated credit scorecard is built on; verified against the log-odds definition.
- [x] M.6 `FeatureSpec` — the train/serve feature contract (columns, order, dtypes) that
  catches or repairs a mismatched serving frame.
- [x] M.7 `feature_profile` — per-column shape plus the transform it is asking for.

## Cluster K — Choosing an operating point, and choosing a model

0.5 is the right cutoff only when the classes are balanced *and* a false positive costs
exactly what a false negative costs. Neither is usually true, and nothing in the metric
surface helped with it.

- [x] K.1 `best_threshold` — the cutoff maximizing F1, F-beta, precision, recall, or
  Youden's J, verified against a brute-force search.
- [x] K.2 `best_cost_threshold` — the cutoff minimizing *expected cost*, given what a false
  positive and a false negative actually cost. The only choice that optimizes the thing you
  care about rather than a proxy; on a 10:1 cost ratio it more than halves the cost of the
  F1-optimal cutoff.
- [x] K.3 `expected_cost_curve` — the whole cost curve, so a flat one is visible before
  anyone argues about the cutoff.
- [x] K.4 `compare_models` — several candidates' metrics in the *same* aggregate, so a
  six-model comparison costs one scan. Returns a `Dataset` that joins to a latency or
  serving-cost column and appends to an experiment log.
- [x] K.5 A rank metric in `compare_models` is refused with a reason rather than silently
  turned into one sort per model.
- [x] K.6 `expected_calibration_error` / `maximum_calibration_error` — the property AUC
  cannot see (does a predicted 0.7 actually occur 70% of the time), exact against a
  reference implementation.
- [x] K.7 `brier_skill_score` — the Brier score rescaled against the base rate so it reads
  like R², exact against scikit-learn's `brier_score_loss`.

## Cluster C — Statistical expressions

- [x] C.1 `plan/functions/analysis/` — the DS statistical toolkit as single-pass `Expr`.
- [x] C.2-C.7 Robust spread: `midhinge`, `trimean`, `quartile_dispersion`, `robust_cv`,
  `interdecile_range`, `decile_ratio`.
- [x] C.8-C.11 Shape: `bowley_skew`, `moors_kurtosis`, `pearson_mode_skew`, `jarque_bera`.
- [x] C.12-C.19 Inference: `group_mean`, `welch_t_statistic`, `welch_df`, `cohens_d`,
  `hedges_g`, `proportion_z_statistic`, `mean_ci_half_width`, `proportion_ci_half_width`.
- [x] C.20-C.22 Association: `point_biserial`, `correlation_ratio`, `signal_ratio`.
- [x] C.23 `ml/stats/` — the statistics that need two passes or a grouping.
- [x] C.24-C.28 `spearman_corr` (average ranks, so it matches SciPy under ties), `entropy`,
  `gini_impurity`, `herfindahl_index`, `mode_share`.
- [x] C.29-C.32 `chi_square`, `cramers_v`, `mutual_information`, `anova_f`.
- [x] C.33-C.36 `trimmed_mean`, `winsorized_mean`, `median_abs_deviation`, `outlier_mask`.
- [x] C.37 **Bug fixed**: the contingency table dropped the cells no row landed in, halving
  chi-squared on a perfectly associated table — reading as "no relationship".
- [x] C.38 Mutual information skips empty cells rather than returning NaN for the whole sum.

## Cluster D — Drift and data monitoring

- [x] D.1 `ml/stats/drift.py` — reference-versus-current comparison, edges always from the
  reference so a shift moves mass between bins rather than moving the bins.
- [x] D.2-D.5 `population_stability_index`, `kl_divergence`, `js_divergence`,
  `categorical_drift` (total variation, no binning needed).
- [x] D.6-D.7 `woe_table`, `information_value` — the scorecard feature-ranking pair.
- [x] D.8 `drift_report` — per-column PSI, JS, mean shift, and null-rate shift, as a
  Dataset ordered by descending PSI so it appends to a monitoring table.
- [x] D.9 **Bug fixed**: a constant reference column produced exactly 0.0 PSI — "no drift"
  for a column that moved from 1.0 to 2.0. Now an actionable error.
- [x] D.10 `ds.ml.drift(reference=...)` — the accessor, defaulting to every numeric column.

## Cluster E — Preprocessors

- [x] E.1 `QuantileTransformer` — rank mapping via a sum of threshold indicators, uniform
  or normal output.
- [x] E.2 **Bug fixed**: the step function reported each step's lower edge, biasing the
  normal output's mean to -0.098 instead of 0. Now the midpoint.
- [x] E.3 `PowerTransformer` — Yeo-Johnson with the maximum-likelihood lambda found by
  evaluating the whole candidate grid *in one aggregate pass*, not one pass per iteration.
- [x] E.4 `yeo_johnson` exposed as a plain expression builder.
- [x] E.5-E.7 `LogTransformer`, `Clipper` (winsorizing at learned quantiles),
  `MissingIndicator` (the flag that must be created before an imputer destroys the signal).
- [x] E.8-E.10 `FrequencyEncoder`, `RareCategoryEncoder`, `HashingEncoder` — the three
  answers to a categorical column whose cardinality breaks a one-hot.
- [x] E.11 `normal_ppf` — Acklam's inverse normal CDF, so a normal-output transform needs
  no SciPy and evaluates no inverse CDF per row.
- [x] E.12 `DateTimeFeaturizer` — a timestamp expanded into calendar parts a tree can split.
- [x] E.13 `CyclicalEncoder` — the periodic parts as `(sin, cos)`, fixing the wrap-around
  that puts hour 23 and hour 0 twenty-three units apart for every distance-based model.
- [x] E.14 `LagFeaturizer` — the value `n` rows back, within each series.
- [x] E.15 `RollingFeaturizer` — an aggregate over the window **before** the current row,
  with no option to include it: a rolling mean containing the current row puts the target's
  own value inside its own feature, which is the commonest forecasting leak and raises
  nothing.
- [x] E.16 `preprocessors/timeseries/` — the two split by where the feature comes from
  (the row itself vs. its neighbours), because only the second can leak.
- [x] E.17 `Binarizer` — threshold a column to 0/1, the feature a linear model can't build.
- [x] E.18 `InteractionFeatures` — pairwise products, the interactions a linear model can't
  learn and a tree finds for free.
- [x] E.19 `RatioFeatures` — the rate a raw pair hides, with a zero denominator becoming
  null (via `nullif`) rather than infinity.
- [x] E.20 `ColumnSelector` / `ColumnDropper` — projection as a pipeline stage, so the exact
  feature set travels with the fitted `Chain`.
- [x] E.21 `VarianceThreshold` — learn which columns are worth keeping on train, apply on
  serve.
- [x] E.22 `GroupStatEncoder` — per-group statistics joined onto every row (the single most
  productive tabular feature family), learned on train so a serving row inherits its group's
  *training* behaviour.
- [x] E.23 `GroupImputer` — fill a null with its group's mean, not the global one, with the
  global mean as the fallback for an unseen group.
- [x] E.24 `preprocessors/derived/` subpackage, so the family grows without breaking the
  ≤12-files-per-directory limit.

## Cluster F — Persistence and interop

- [x] F.1 `ml/preprocessors/persistence.py` — `to_dict` / `from_dict` / `save` / `load`.
- [x] F.2 JSON rather than pickle: reviewable, diffable, portable, and safe to load.
- [x] F.3 Tuples and non-string dict keys survive the round trip (JSON would lose both).
- [x] F.4 A versioned schema, so a document from a future build is an error not a misread.
- [x] F.5 The class registry is derived from the package's own `__all__`, so a new
  preprocessor is loadable the moment it is exported.
- [x] F.6 `Preprocessor.save` / `Preprocessor.load` methods, cloud URIs included.
- [x] F.7 Round-trip parity test: a reloaded preprocessor must transform *identically*.
- [x] F.8 `FeatureSpec` — the train/serve feature contract (columns, order, dtypes) that
  catches or repairs the wrong-order / extra-column / retyped serving frame a model would
  otherwise score against silently. `validate` raises, `align` repairs, and it saves as JSON.

## Cluster G — Splitting, sampling, cross-validation

- [x] G.1 `ml/splitting.py` — folds as content-hash filters, never a materialized shuffle.
- [x] G.2 `fold_column` — the primitive, so a split can be written to disk and reused.
- [x] G.3-G.6 `kfold`, `stratified_kfold`, `group_kfold`, `time_series_split`
  (expanding or rolling window).
- [x] G.7 `ds.ml.kfold(k, stratify=..., group=...)` — one method, the variant selected by
  the constraint that matters, with the two mutually-exclusive options guarded.
- [x] G.8 `ds.ml.time_series_split(...)`.

## Cluster H — Model-free feature selection

- [x] H.1 `ml/selection.py` — deciding what to keep without fitting a model first, which is
  both cheaper and less circular than reading a model's own importances.
- [x] H.2 `constant_columns` — zero variance, and the more insidious near-constant case.
- [x] H.3 `correlated_columns` — a deterministic pruning rule, so two runs agree on which of
  two identical columns survived.
- [x] H.4 `feature_report` — every candidate ranked by information value, point-biserial
  correlation, class separation, and null rate, as one sortable `Dataset`.
- [x] H.5 `feature_profile` — what each column *is* before any target is involved (null
  rate, concentration, skew, tail weight) plus the transform its shape is asking for.
  `Dataset.profile` answers the data-quality question; this answers the modelling one.

## Cluster J — Repo hygiene found along the way

- [x] J.1 `tools/lint_structure.py`: `STRUCTURE_ALLOW` now covers `__init__.py`, which the
  documented escape hatch always implied but the checker did not honor.
- [x] J.2 A found bug in `RatioFeatures`' first draft — `lit(None)` is unsupported; the fix
  uses `nullif`, which is the correct spelling and taught nothing wrong to callers.

## Cluster I — Docs, tests, examples

- [x] I.1 `tests/unit/test_ml_tabular_predict.py` — 44 tests.
- [x] I.2 `tests/unit/test_ml_metrics.py` — 57 tests, scikit-learn parity throughout.
- [x] I.3 `tests/unit/test_ml_preprocessors_ds.py` — 42 tests.
- [x] I.4 `tests/unit/test_ml_stats_drift.py` — 54 tests, SciPy parity throughout.
- [x] I.5 `tests/unit/test_ml_ranking_selection.py` — 37 tests against hand-written
  reference implementations.
- [x] I.6 Runnable doctests on every new public name (159 of them).
- [x] I.7 `docs/ml/tabular-models.md` — the GBDT/sklearn/ONNX scoring guide.
- [x] I.8 `docs/ml/evaluation.md` — metrics, per-segment scoring, diagnostic tables.
- [x] I.9 `docs/ml/statistics-and-drift.md` — statistics, drift, cross-validation.
- [x] I.10 `docs/ml/preprocessors.md` extended: distribution shaping, high-cardinality
  encoding, timestamp features, persistence.
- [x] I.11 `docs/api/ml.md` — reference sections for tabular, metrics, ranking, stats,
  splitting, selection, and persistence.
- [x] I.12 `docs/ml/index.md` — a new "Measure what the model does" section and toctree.
- [x] I.13 `examples/tabular_ml.py` — the whole lifecycle as one runnable script.
- [x] I.14 `docs/ml/evaluation.md` extended with the multi-class section;
  `docs/ml/preprocessors.md` with timestamp features, lag/rolling history, and persistence.

---

## Verification

- 736 test functions across the new test files, all green, with scikit-learn and SciPy as
  oracles where one exists and hand-written reference implementations where none does.
- ~315 new executed doctests.
- **300 new public names** across thirty-two new packages.
- Every metric that scikit-learn also defines is checked against it at 1e-12; every one it
  does not is checked against a closed-form identity or a naive reference implementation.
- `ruff check` / `ruff format` clean, `lint-structure` clean, all five import-linter layer
  contracts kept, `MAP.md` current, the whole `tests/docs` suite green.
- Four real correctness bugs found and fixed while building this, each of which produced a
  **plausible wrong number** rather than an error: the contingency table's missing empty
  cells, the constant-reference drift silently reading 0.0, the quantile transform's
  lower-edge bias, and float32 features shifting a float64 estimator's output.
