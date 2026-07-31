# Statistics, drift, and cross-validation

This page covers the analysis surface around a model rather than the model itself: the statistics that tell you whether a feature is worth having, the drift measures that tell you whether it still is, and the splits that make a validation score trustworthy.

## Statistical expressions

The single-pass statistics are `Expr`, so they belong inside `agg()` and compose with `group_by` exactly as the metrics do:

```python
import batcher as bt

ds = bt.from_pydict({"latency": [12.0, 15.0, 18.0, 22.0, 4000.0]})
print(ds.agg(median=bt.col("latency").median(), robust=bt.trimean("latency")).to_pydict())
```

### Robust spread

The mean and the standard deviation are the wrong summary for most real columns, because a single bad row moves both without limit. These are built from quantiles instead:

| Function | What it measures |
|---|---|
| `midhinge` | The midpoint of the middle half, ignoring the outer quartiles entirely. |
| `trimean` | Tukey's robust location estimate, weighting the median twice. |
| `quartile_dispersion` | Unitless spread in `[0, 1]`, comparable across columns. |
| `robust_cv` | Interquartile range over the median: the outlier-proof coefficient of variation. |
| `interdecile_range` | The span containing the middle 80% of values. |
| `decile_ratio` | P90 over P10, the classic inequality ratio. |

A second family expresses spread *relative to level*, so the number is unitless and comparable across columns on different scales. `bt.index_of_dispersion` is the variance-to-mean ratio (the Fano factor, exactly 1 for a Poisson process), `bt.signal_to_noise` is the mean over the standard deviation (the reciprocal of the coefficient of variation), `bt.studentized_range` is the range in standard deviations (a quick outlier smell), and `bt.relative_range` is the range over the mean. Each is a single aggregate over the existing moment primitives:

```python
ds = bt.from_pydict({"counts": [8.0, 12.0, 9.0, 11.0, 10.0]})
print(ds.agg(fano=bt.index_of_dispersion("counts"), snr=bt.signal_to_noise("counts")).to_pydict())
```

`bt.geometric_std` is the multiplicative standard deviation for a strictly positive, log-normal column that spans orders of magnitude: a value of 2 means a typical observation is within a factor of 2 of the geometric mean, which describes scatter on a log scale honestly where an ordinary standard deviation is dominated by the largest values.

### Distribution shape

Whether a column is symmetric and how heavy its tails are decides which model and which transform are appropriate:

```python
ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})
print(ds.agg(skew=bt.bowley_skew("x"), normality=bt.jarque_bera("x")).to_pydict())
```

`bowley_skew` and `moors_kurtosis` are the quantile-based versions, which stay meaningful on a column whose fourth moment does not exist. That covers most real latency, revenue, and file-size columns. `jarque_bera` combines skew and kurtosis into the standard normality statistic, useful as a screen over hundreds of features.

### Weighted statistics

Survey weights, recency decay, and per-group sizes all give some rows more influence than others,
and the plain mean and variance are wrong once they do. `bt.weighted_mean`, `bt.weighted_var`, `bt.weighted_std`,
`bt.weighted_covariance`, and `bt.weighted_correlation` are the frequency-weighted forms, each
a single aggregate matching `numpy.average`:

```python
survey = bt.from_pydict({"income": [30.0, 80.0, 55.0], "weight": [3.0, 1.0, 2.0]})
print(survey.agg(m=bt.weighted_mean("income", "weight")).to_pydict())
```

### Two-sample comparison

An A/B test or a cohort comparison is arithmetic over *conditional* aggregates, so both samples are summarized in one pass and neither leaves the engine:

```python
ds = bt.from_pydict(
    {"value": [10.0, 11.0, 12.0, 20.0, 21.0, 22.0], "arm": ["a"] * 3 + ["b"] * 3}
)
arm_a = bt.col("arm") == bt.lit("a")
print(
    ds.agg(
        t=bt.welch_t_statistic("value", arm_a),
        df=bt.welch_df("value", arm_a),
        effect=bt.cohens_d("value", arm_a),
    ).to_pydict()
)
```

`welch_t_statistic` is the unequal-variance test, which is the one to use by default. `cohens_d` and `hedges_g` give the effect *size*, which is what distinguishes a real effect from a merely detectable one. At a large enough row count every difference is "significant".

`proportion_z_statistic` is the conversion-rate equivalent, and `mean_ci_half_width` / `proportion_ci_half_width` give the error bar. `group_mean` is the building block all of them share: the mean of a column over the rows a boolean expression selects. Reach for it directly whenever you want one arm's average without running a second query.

### Screening features against a target

Four measures answer "is this feature worth keeping", each for a different pair of types:

```python
ds = bt.from_pydict(
    {"tenure": [1.0, 2.0, 8.0, 9.0], "churned": [False, False, True, True]}
)
churned = bt.col("churned")
print(
    ds.agg(
        correlation=bt.point_biserial("tenure", churned),
        separation=bt.signal_ratio("tenure", churned),
    ).to_pydict()
)
```

`point_biserial` is Pearson's correlation with a boolean coded 0/1, so a numeric feature and a boolean one rank on the same `[-1, 1]` axis. `signal_ratio` asks only whether the feature *separates* the two classes, in standard deviations, so it survives a relationship that reverses direction and a correlation would miss.

`correlation_ratio` is the categorical-feature version: the share of a numeric column's variance that sits *between* groups rather than within them. It takes the per-row group mean, which is what keeps it a single aggregate:

```python
ds = bt.from_pydict({"spend": [1.0, 2.0, 10.0, 11.0], "plan": ["free", "free", "pro", "pro"]})
with_means = ds.with_columns(m=bt.mean(bt.col("spend")).over(partition_by=["plan"]))
print(round(with_means.agg(eta=bt.correlation_ratio("spend", "m")).to_pydict()["eta"][0], 4))
```

`pearson_mode_skew` reads directly as "how many standard deviations the average sits above the most common value", which is the sentence a non-statistician understands. Reach for it when the audience for a data-quality report is not the modelling team.

```{note}
These expressions return a statistic, not a p-value. To turn one into a decision, use the matching test in `batcher.ml.stats` (the hypothesis tests described later on this page), which pairs the statistic with a dependency-free p-value on the driver. That p-value is arithmetic on the one aggregated number, not a second pass over the data.
```

### Statistics that need a second pass

A rank correlation needs an ordering and a trimmed mean needs the quantiles before it can filter on them, so these are functions over a `Dataset` rather than expressions. They are still entirely relational, built from a window, a `group_by`, or a second aggregate, so nothing materializes on the driver:

```python
from batcher.ml.stats import cramers_v, entropy, mutual_information, spearman_corr

ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 4.0, 9.0, 16.0]})
print(round(spearman_corr(ds, "x", "y"), 6))

cats = bt.from_pydict({"a": ["x", "x", "y", "y"], "b": ["p", "p", "q", "q"]})
print(entropy(cats, "a"), cramers_v(cats, "a", "b"), mutual_information(cats, "a", "b"))
```

`spearman_corr` sees a monotone relationship a Pearson correlation underrates, and is immune to outliers because an extreme value contributes only its rank. `cramers_v` is the categorical counterpart of a correlation and, unlike `chi_square`, does not grow with the row count, so it ranks features consistently across datasets of different sizes.

`correlation_matrix` and `covariance_matrix` give the whole pairwise structure of a feature set in one scan, returned as a labeled square `Dataset`. Reading down a column shows what a feature moves with, which is what flags redundant features and multicollinearity. `partial_correlation` goes one step further and removes a confounder: two features can correlate only because both track a third, and the partial correlation is what survives holding that third fixed. `variance_inflation_factor` puts a number on that multicollinearity per feature, measuring how much the rest of the set inflates each column's variance. A VIF above 5 or 10 flags a feature whose linear-model coefficient will be unstable.

Where `cramers_v` is symmetric, `theils_u` is directional: it reports the fraction of one categorical column's uncertainty that knowing the other removes, so `theils_u(ds, "x", "y")` and `theils_u(ds, "y", "x")` differ and answer "does `x` predict `y`" rather than "are they related". For a numeric column against a grouping, `eta_squared` and its bias-corrected sibling `epsilon_squared` are the bounded effect sizes `anova_f` lacks: both read as "this grouping explains 30% of the variance" and stay comparable across sample sizes, which a raw F never is. `omega_squared` corrects the bias furthest for generalizing beyond the sample, and `cohens_f` is the effect-size scale a power analysis is specified on.

`trimmed_mean`, `winsorized_mean`, `median_abs_deviation`, and `outlier_mask` cover robust location and outlier detection. The `|x - median| / MAD > 3` rule that `outlier_mask` implements is what to use instead of a z-score on anything with a tail.

## Profiling features before modeling

`Dataset.profile` answers the data-quality question of how much is present and how many distinct
values there are. `feature_profile` answers the modeling one in the same single pass, and names the
transform each column is asking for:

```python
from batcher.ml.selection import feature_profile

ds = bt.from_pydict(
    {"flat": [1.0] * 8, "skewed": [float(2**i) for i in range(8)], "ok": [float(i) for i in range(8)]}
)
print(feature_profile(ds).sort("column").to_pydict()["suggestion"])
```

`constant_columns` and `correlated_columns` are the two pruning screens, both model-free.
The second uses a deterministic rule for which of a redundant pair to drop, because a screen
that depends on iteration order gives a different feature set on every run.

`feature_report` ranks every candidate against a binary target by information value,
point-biserial correlation, class separation, and null rate. Those four numbers catch
different kinds of signal, so a feature strong on any of them survives.

`batcher.ml.feature_scores` is the univariate filter that scikit-learn's `SelectKBest` runs, one score per feature against the target: `f_classif_scores` (ANOVA F for a categorical target), `f_regression_scores` (regression F for a continuous one), `chi2_scores` (categorical against categorical), and `mutual_info_scores` (bits shared, which catches a non-monotone link the F scores miss). `select_k_best` turns any of those score dicts into the columns to keep.

```python
from batcher.ml.feature_scores import f_classif_scores, select_k_best

ds = bt.from_pydict(
    {"y": ["a", "a", "b", "b"], "signal": [1.0, 1.1, 9.0, 9.2], "noise": [5.0, 1.0, 5.0, 1.0]}
)
print(select_k_best(f_classif_scores(ds, "y"), 1))
```

A univariate score sees a feature that only matters in combination with another as noise, so use it to prune obvious dead weight, not as the last word on a feature set.

## Explaining a model

Once a model is trained, `batcher.ml.interpret` says *why* it predicts what it does. It answers over
the whole dataset, because it re-scores through the engine rather than on a driver sample.

`permutation_importance` ranks features by how far the error rises when each is shuffled. It
is model-agnostic and honest in a way a tree's built-in importance is not: a tree can call a
feature important because it split on it, even when permuting the feature changes nothing.

```python
# docs: skip
from batcher.ml.interpret import partial_dependence, permutation_importance

predict = lambda d: d.ml.predict(model, features=feature_names)
importance = permutation_importance(test, predict, feature_names, y_true="label")
```

`partial_dependence` traces what the model does as one feature varies, averaged over the
real joint distribution of the others. That is the curve a stakeholder reads as "risk rises with
balance, then plateaus".

## Outlier detection

`batcher.ml.outliers` finds the rows that do not belong to the same process as the rest. The
rule is the choice, and there are three, from least to most robust: `zscore` (mean-based, wrong
on a skewed column), `iqr` (Tukey's fence, the distribution-free default), and `mad` (the most
robust, for a heavy tail).

```python
from batcher.ml.outliers import count_outliers, flag_outliers

ds = bt.from_pydict({"latency": [10.0, 12.0, 11.0, 13.0, 5000.0]})
print(count_outliers(ds, "latency", method="iqr"))
flagged = flag_outliers(ds, "latency", method="iqr")   # a boolean flag column, not a drop
```

`flag_outliers` marks them (the decision to keep or drop is yours), `count_outliers` tallies
them, and `OutlierClipper` clamps them as a fitted preprocessor that applies the *training*
bounds to serving data. `outlier_bounds` returns the raw cut points.

Those rules are univariate, judging one column at a time. `mahalanobis_distance` is the multivariate score for a row that looks ordinary on every column but is an outlier in the *joint* distribution, measuring distance from the center in units that account for the columns' correlations. Its square is chi-squared with one degree of freedom per column, which is how you turn it into a threshold.

```python
from batcher.ml.outliers import mahalanobis_distance

ds = bt.from_pydict({"height": [60.0, 65.0, 70.0, 62.0], "weight": [120.0, 150.0, 180.0, 40.0]})
scored = mahalanobis_distance(ds, ["height", "weight"])
print(scored.to_pydict()["mahalanobis"][3] == max(scored.to_pydict()["mahalanobis"]))  # the light-but-average-height row
```

## Drift monitoring

When a model has been deployed and the labels have not arrived yet, the only observable thing is whether the *inputs* still look like the training data.

```python
from batcher.ml.stats import drift_report, population_stability_index

train = bt.from_pydict({"x": [float(i) for i in range(200)]})
today = bt.from_pydict({"x": [float(i) + 60 for i in range(200)]})
print(round(population_stability_index(train, today, "x", buckets=5), 4))
```

The bin edges always come from the **reference** distribution and are then applied unchanged to the current data, so a shift shows up as mass moving between bins rather than as the bins themselves moving. Deriving edges separately for each side would make two very different distributions look identical.

Read the numbers with the conventions the monitoring literature settled on:

| Measure | Reading |
|---|---|
| `population_stability_index` | Below 0.1 stable; 0.1 to 0.25 moderate; above 0.25 significant. |
| `js_divergence` | 0 identical, 1 bit maximally different. Comparable across columns. |
| `kl_divergence` | Asymmetric: punishes current mass where the reference had almost none. |
| `categorical_drift` | The share of mass that would have to move for the two to match. |
| `information_value` | Below 0.02 useless; 0.02 to 0.1 weak; 0.1 to 0.3 medium; above 0.3 strong. |

`drift_report` runs the whole check and returns a `Dataset` ordered by descending PSI, which is what makes it appendable to a monitoring table. A single PSI is far less informative than its history:

```python
report = drift_report(train, today, ["x"], buckets=5)
print(report.columns)
```

`woe_table` and `information_value` are the scorecard pair: bin a feature and report the log odds of a positive in each bin. A monotone WOE column is what makes a feature usable in a linear scorecard, and the shape of the table tells you where to merge bins.

## Resampling for imbalanced learning

A classifier trained on a 1%-positive dataset learns to predict "negative" and scores well on
accuracy while being useless. `batcher.ml.sampling` reshapes the class balance as a relational
operation: an exact content-hashed filter or concatenation, never a driver-side shuffle. That is
what lets it run over a dataset larger than memory.

```python
import batcher as bt
from batcher.ml.sampling import class_counts, class_weights, oversample, undersample

ds = bt.from_pydict({"y": [0] * 100 + [1] * 10, "x": list(range(110))})
print(class_counts(undersample(ds, "y"), "y"))   # exactly balanced by discarding
print(class_counts(oversample(ds, "y"), "y"))     # exactly balanced by duplicating
```

`undersample` discards majority rows; `oversample` duplicates minority rows deterministically;
`balanced_sample` moves every class to the median count. When the model supports it, prefer
`class_weights` (a `{class: weight}` dict for the model's ``class_weight``) or `sample_weights`
(a per-row weight column). Both rebalance the *loss* without discarding or duplicating a
single row. `class_counts` is the first thing to look at.

`stratified_sample` is the different tool for a different job: it keeps the same fraction of *every* stratum rather than equalizing them, so it shrinks a dataset for a quick experiment while preserving its class balance. You get a proportional 10% sample rather than 10% of the whole, which would starve the rare classes.

```python
from batcher.ml.sampling import class_counts, stratified_sample

ds = bt.from_pydict({"y": [0] * 100 + [1] * 20, "x": list(range(120))})
print(class_counts(stratified_sample(ds, "y", 0.5, seed=1), "y"))  # {0: 50, 1: 10}
```

## Cross-validation splits

A fold here is a **content hash** of each row compared against fold boundaries, never a materialized shuffle. That means a fold is an ordinary row-wise filter, the assignment is identical however the data is partitioned, and the training half of a fold stays lazy until something reads it.

```python
ds = bt.range(0, 1000)
folds = ds.ml.kfold(5, key="value")
print(sum(validate.count() for _, validate in folds))
```

Two options select the variant your data needs, and choosing correctly is usually the difference between a trustworthy score and a misleading one:

`stratify=` keeps each label's proportion identical in every fold. Use it whenever the label is imbalanced, or the fold-to-fold variance in the score measures the split rather than the model.

```python
ds = bt.from_pydict({"y": [0] * 90 + [1] * 10, "x": list(range(100))})
folds = ds.ml.kfold(5, key="x", stratify="y")
print([v.filter(bt.col("y") == 1).count() for _, v in folds])
```

`group=` keeps every row of a group in the same fold. Use it whenever rows repeat an entity such as a user, a patient, a session, or a document. Without it the model memorizes the entity rather than the pattern, cross-validation looks excellent, and production does not. This is the most common silent leak in applied ML.

For a time series, neither applies: a random fold puts next week's rows in the training set, so the model sees the future and the validation score is one no deployment will reproduce.

```python
ds = bt.from_pydict({"t": list(range(100)), "x": list(range(100))})
print([(train.count(), validate.count()) for train, validate in ds.ml.time_series_split("t", 4)])
```

`expanding=True` (the default) grows the training window with each split, matching a model retrained on all history; `expanding=False` slides a fixed-width window, matching one that deliberately forgets.

`batcher.ml.model_selection` runs the loop end to end: `cross_val_score` fits and scores a
model on each fold (the spread across folds is the honesty a single number hides),
`cross_val_predict` gives every row its out-of-fold prediction (the unbiased input a stacking
ensemble needs), and `learning_curve` scores against training-set size to answer whether more
data would help. Each takes a `fit` and a `predict` callable, so any scikit-learn-style model
composes.

`batcher.ml.splitting.fold_column` is the primitive underneath. Reach for it when the split should outlive the pipeline that created it: it writes one column that every downstream job can filter on without re-deriving the assignment.

## Hypothesis tests

A test statistic says how large an effect is; the p-value says how surprising it is under the null hypothesis, and the p-value is what you act on. `batcher.ml.stats` pairs each statistic with its p-value in one pass and returns a `TestResult` carrying the statistic, its degrees of freedom, and the p-value.

Use `t_test_1samp` to check a column's mean against a target, `t_test_ind` for Welch's two-sample test of two groups, `anova_test` to extend that to several groups, `chi_square_test` for the independence of two categorical columns, and `normality_test` (Jarque-Bera) to screen a column before assuming it is Gaussian. `pearson_test` and `spearman_test` add a p-value to a linear or monotone correlation, `proportion_ztest` checks a success rate against a target (with `binomial_test` the exact small-sample version), and `mcnemar_test` compares two classifiers' error rates on the same rows. `mcnemar_test` is the paired test to reach for when deciding whether one model genuinely beats another.

```python
import batcher as bt
from batcher.ml.stats import t_test_ind, anova_test

ds = bt.from_pydict(
    {"g": ["a", "a", "a", "b", "b", "b"], "x": [1.0, 2.0, 3.0, 8.0, 9.0, 10.0]}
)
result = t_test_ind(ds, "x", "g")
print(round(result.pvalue, 4), result.pvalue < 0.05)
```

`bartlett_test` and `levene_test` check the equal-variance assumption a t-test and an ANOVA quietly rely on. Bartlett's is the powerful choice for normal groups, and Levene's (median-centered) is the robust default.

When the data itself is too skewed or ordinal for a t-test, `mann_whitney_u` (two groups) and `kruskal_wallis` (several) are the rank-based, distribution-free alternatives, asking whether one group tends to larger ranks rather than a larger mean. For *paired* measurements such as a before/after or matched-pair design, `wilcoxon_signed_rank` is the distribution-free replacement for the paired t-test. `friedman_test` extends that to several treatments measured on the same blocks, giving you the non-parametric repeated-measures ANOVA.

Report `cliffs_delta` or `common_language_effect_size` beside a Mann-Whitney result. The test says *whether* two groups differ; these say *how much*, as the probability that a random member of one exceeds a random member of the other.

The tail probabilities come from dependency-free implementations of the Student's t, F, and chi-squared survival functions, checked against SciPy. The whole reduction is a handful of aggregates, so a test scales the same way every other statistic here does.

## Time-series diagnostics

A time series carries its signal in how a column relates to its own past, which a Pearson correlation cannot see. `batcher.ml.timeseries` orders a column by a time key and measures that self-relationship. `autocorrelation` gives the lag-`k` value, `autocorrelations` the whole function up to a maximum lag, `ljung_box` pools the first several lags into one white-noise test, and `durbin_watson` is the regression diagnostic for autocorrelated residuals. `partial_autocorrelation` and `partial_autocorrelations` give the *partial* function, which strips out what the intervening lags already explain and so cuts off sharply at the order of an autoregressive process. That sharp cutoff is what makes it the tool for choosing the order.

```python
import batcher as bt
from batcher.ml.timeseries import autocorrelations, ljung_box

ds = bt.from_pydict({"t": list(range(12)), "sales": [float(i % 4) for i in range(12)]})
print({k: round(v, 3) for k, v in autocorrelations(ds, "sales", 4, order_by="t").items()})
print(ljung_box(ds, "sales", 4, order_by="t").pvalue < 0.05)
```

For scoring a forecast, `mean_absolute_scaled_error` is the scale-free metric: the model's mean absolute error divided by the naive seasonal forecast's, so a value below 1 beats naive and the number is comparable across series on any scale.

An autocorrelation needs the whole series in time order, so unlike the mergeable statistics above these run over a single ordered window rather than a partitionable aggregate. The formulas are the Box-Jenkins definitions, checked against independent numpy references.

## Requirements and limitations

A drift measure needs a reference column with more than one distinct value. A constant reference raises rather than reporting 0.0, because "no drift" for a column that moved from 1.0 to 2.0 is the worst possible answer.

`js_divergence` does not reach 1 for a wholly shifted column, because the outermost reference bins are open-ended and absorb everything beyond them. Alert on `population_stability_index`, which has no such ceiling; use JS to compare across columns.

Fold sizes are binomial around `n / k` rather than exact, as with any hash-keyed split, and `group_kfold`'s folds vary further because groups differ in size.

## See also

- {doc}`/ml/evaluation/evaluation`: score a model once you have a trustworthy split.
- {doc}`/ml/preparing/preprocessors/index`: the transforms these statistics tell you a column needs.
- {doc}`/user-guide/trust/data-quality`: assert contracts rather than measure them.
- {doc}`/cookbook/statistics/index`: short runnable recipes for the functions on this page.
