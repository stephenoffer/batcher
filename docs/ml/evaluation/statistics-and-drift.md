# Statistics and drift

This page covers the analysis surface around a model rather than the model itself: the statistics and profiles that tell you whether a feature is worth having, the outlier rules and hypothesis tests you judge them with, and the drift measures that tell you whether they still hold once the model is deployed.

## Statistical expressions

The single-pass statistics are {py:class}`Expr <batcher.plan.expr_ir.core.Expr>`, so they belong inside `agg()` and compose with `group_by` exactly as the metrics do:

```python
import batcher as bt

ds = bt.from_pydict({"latency": [12.0, 15.0, 18.0, 22.0, 4000.0]})
print(ds.agg(median=bt.col("latency").median(), robust=bt.trimean("latency")).to_pydict())
```

### Robust spread

The mean and the standard deviation are the wrong summary for most real columns. A single bad row moves both without limit. These are built from quantiles instead:

| Function | What it measures |
|---|---|
| `midhinge` | The midpoint of the middle half, ignoring the outer quartiles entirely. |
| `trimean` | Tukey's robust location estimate, weighting the median twice. |
| {py:func}`quartile_dispersion <batcher.quartile_dispersion>` | Unitless spread in `[0, 1]`, comparable across columns. |
| {py:func}`robust_cv <batcher.robust_cv>` | Interquartile range over the median: the outlier-proof coefficient of variation. |
| {py:func}`interdecile_range <batcher.interdecile_range>` | The span containing the middle 80% of values. |
| {py:func}`decile_ratio <batcher.decile_ratio>` | P90 over P10, the classic inequality ratio. |

A second family expresses spread *relative to level*, so the number is unitless and comparable across columns on different scales. {py:func}`bt.index_of_dispersion <batcher.index_of_dispersion>` is the variance-to-mean ratio (the Fano factor, exactly 1 for a Poisson process), {py:func}`bt.signal_to_noise <batcher.signal_to_noise>` is the mean over the standard deviation (the reciprocal of the coefficient of variation), {py:func}`bt.studentized_range <batcher.studentized_range>` is the range in standard deviations (a quick outlier smell), and {py:func}`bt.relative_range <batcher.relative_range>` is the range over the mean. Each is a single aggregate over the existing moment primitives:

```python
ds = bt.from_pydict({"counts": [8.0, 12.0, 9.0, 11.0, 10.0]})
print(ds.agg(fano=bt.index_of_dispersion("counts"), snr=bt.signal_to_noise("counts")).to_pydict())
```

{py:func}`bt.geometric_std <batcher.geometric_std>` is the multiplicative standard deviation for a strictly positive, log-normal column that spans orders of magnitude. A value of 2 means a typical observation sits within a factor of 2 of the geometric mean. That describes scatter on a log scale honestly, where an ordinary standard deviation is dominated by the largest values.

### Distribution shape

Whether a column is symmetric and how heavy its tails are decides which model and which transform are appropriate:

```python
ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})
print(ds.agg(skew=bt.bowley_skew("x"), normality=bt.jarque_bera("x")).to_pydict())
```

{py:func}`bowley_skew <batcher.bowley_skew>` and {py:func}`moors_kurtosis <batcher.moors_kurtosis>` are the quantile-based versions, which stay meaningful on a column whose fourth moment does not exist. That covers most real latency, revenue, and file-size columns. {py:func}`jarque_bera <batcher.jarque_bera>` combines skew and kurtosis into the standard normality statistic, useful as a screen over hundreds of features.

### Weighted statistics

Survey weights, recency decay, and per-group sizes all give some rows more influence than others,
and the plain mean and variance are wrong once they do. {py:func}`bt.weighted_mean <batcher.weighted_mean>`, {py:func}`bt.weighted_var <batcher.weighted_var>`, {py:func}`bt.weighted_std <batcher.weighted_std>`,
{py:func}`bt.weighted_covariance <batcher.weighted_covariance>`, and {py:func}`bt.weighted_correlation <batcher.weighted_correlation>` are the frequency-weighted forms, each
a single aggregate matching `numpy.average`:

```python
survey = bt.from_pydict({"income": [30.0, 80.0, 55.0], "weight": [3.0, 1.0, 2.0]})
print(survey.agg(m=bt.weighted_mean("income", "weight")).to_pydict())
```

Nulls are dropped pairwise. A row missing the value, the weight, or either side of a covariance leaves every sum of that aggregate together, which is what `numpy.average` over the complete rows computes. The row with the missing income below contributes nothing, not even its weight:

```python
gaps = bt.from_pydict({"income": [1.0, None, 3.0], "weight": [1.0, 1.0, 1.0]})
assert gaps.agg(v=bt.weighted_var("income", "weight")).to_pydict() == {"v": [1.0]}
```

### Two-sample comparison

An A/B test or a cohort comparison is arithmetic over *conditional* aggregates, so both samples are summarized in one pass and neither leaves the engine:

```python
ds = bt.from_pydict({"value": [10.0, 11.0, 12.0, 20.0, 21.0, 22.0], "arm": ["a"] * 3 + ["b"] * 3})
arm_a = bt.col("arm") == bt.lit("a")
print(
    ds.agg(
        t=bt.welch_t_statistic("value", arm_a),
        df=bt.welch_df("value", arm_a),
        effect=bt.cohens_d("value", arm_a),
    ).to_pydict()
)
```

{py:func}`welch_t_statistic <batcher.welch_t_statistic>` is the unequal-variance test. Use it by default. {py:func}`cohens_d <batcher.cohens_d>` and {py:func}`hedges_g <batcher.hedges_g>` give the effect *size*, which is what distinguishes a real effect from a merely detectable one. At a large enough row count every difference is "significant".

{py:func}`proportion_z_statistic <batcher.proportion_z_statistic>` is the conversion-rate equivalent, and {py:func}`mean_ci_half_width <batcher.mean_ci_half_width>` / {py:func}`proportion_ci_half_width <batcher.proportion_ci_half_width>` give the error bar. {py:func}`group_mean <batcher.group_mean>` is the building block all of them share: the mean of a column over the rows a boolean expression selects. Reach for it directly whenever you want one arm's average without running a second query.

### Screening features against a target

Four measures answer "is this feature worth keeping", each for a different pair of types:

```python
ds = bt.from_pydict({"tenure": [1.0, 2.0, 8.0, 9.0], "churned": [False, False, True, True]})
churned = bt.col("churned")
print(
    ds.agg(
        correlation=bt.point_biserial("tenure", churned),
        separation=bt.signal_ratio("tenure", churned),
    ).to_pydict()
)
```

{py:func}`point_biserial <batcher.point_biserial>` is Pearson's correlation with a boolean coded 0/1, so a numeric feature and a boolean one rank on the same `[-1, 1]` axis. {py:func}`signal_ratio <batcher.signal_ratio>` asks only whether the feature *separates* the two classes, in standard deviations, so it survives a relationship that reverses direction and a correlation would miss.

{py:func}`correlation_ratio <batcher.correlation_ratio>` is the categorical-feature version: the share of a numeric column's variance that sits *between* groups rather than within them. It takes the per-row group mean, which is what keeps it a single aggregate:

```python
ds = bt.from_pydict({"spend": [1.0, 2.0, 10.0, 11.0], "plan": ["free", "free", "pro", "pro"]})
with_means = ds.with_columns(m=bt.mean(bt.col("spend")).over(partition_by=["plan"]))
print(round(with_means.agg(eta=bt.correlation_ratio("spend", "m")).to_pydict()["eta"][0], 4))
```

{py:func}`pearson_mode_skew <batcher.pearson_mode_skew>` reads directly as "how many standard deviations the average sits above the most common value", which is the sentence a non-statistician understands. Reach for it when the audience for a data-quality report is not the modelling team.

```{note}
These expressions return a statistic, not a p-value. To turn one into a decision, use the matching test in `batcher.ml.stats` (the hypothesis tests described later on this page), which pairs the statistic with a dependency-free p-value on the driver. That p-value is arithmetic on the one aggregated number, not a second pass over the data.
```

### Statistics that need a second pass

A rank correlation needs an ordering and a trimmed mean needs the quantiles before it can filter on them, so these are functions over a {py:class}`Dataset <batcher.Dataset>` rather than expressions. They live in the `batcher.ml.stats` module and are not re-exported as `bt.*`, so import them from there and pass the dataset as the first argument. They are still entirely relational, built from a window, a {py:meth}`group_by <batcher.Dataset.group_by>`, or a second aggregate, so nothing materializes on the driver:

```python
from batcher.ml.stats import cramers_v, entropy, mutual_information, spearman_corr

ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 4.0, 9.0, 16.0]})
print(round(spearman_corr(ds, "x", "y"), 6))

cats = bt.from_pydict({"a": ["x", "x", "y", "y"], "b": ["p", "p", "q", "q"]})
print(entropy(cats, "a"), cramers_v(cats, "a", "b"), mutual_information(cats, "a", "b"))
```

`spearman_corr` sees a monotone relationship a Pearson correlation underrates. How extreme an outlier is does not matter. It contributes only its rank. `cramers_v` is the categorical counterpart of a correlation. Unlike `chi_square` it does not grow with the row count, so it ranks features consistently across datasets of different sizes.

`correlation_matrix` and `covariance_matrix` give the whole pairwise structure of a feature set in one scan, returned as a labeled square `Dataset`. Reading down a column shows what a feature moves with, which flags redundant features.

`partial_correlation` removes a confounder. Two features can correlate only because both track a third, and the partial correlation is what survives holding that third fixed. `variance_inflation_factor` puts a number on multicollinearity per feature: how much the rest of the set inflates each column's variance. A VIF above 5 or 10 flags a feature whose linear-model coefficient will be unstable.

Where `cramers_v` is symmetric, `theils_u` is directional: it reports the fraction of one categorical column's uncertainty that knowing the other removes, so `theils_u(ds, "x", "y")` and `theils_u(ds, "y", "x")` differ and answer "does `x` predict `y`" rather than "are they related". For a numeric column against a grouping, `eta_squared` and its bias-corrected sibling `epsilon_squared` are the bounded effect sizes `anova_f` lacks: both read as "this grouping explains 30% of the variance" and stay comparable across sample sizes, which a raw F never is. `omega_squared` corrects the bias furthest for generalizing beyond the sample, and `cohens_f` is the effect-size scale a power analysis is specified on.

`trimmed_mean`, `winsorized_mean`, `median_abs_deviation`, and `outlier_mask` cover robust location and outlier detection. The `|x - median| / MAD > 3` rule that `outlier_mask` implements is what to use instead of a z-score on anything with a tail. `mean_abs_deviation` sits between the standard deviation and the MAD: it keeps the mean as its center but weights every deviation linearly, so one outlier moves it far less than it moves the standard deviation.

`normalized_entropy` is `entropy` divided by the entropy of the same number of equally likely values, so it lands in `[0, 1]` whatever the cardinality: 0 for a constant column, 1 for a uniform one. Use it rather than raw `entropy` to rank or threshold columns with different numbers of categories:

```python
from batcher.ml.stats import mean_abs_deviation, normalized_entropy

spread = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0]})
print(mean_abs_deviation(spread, "x"))  # 1.0
print(normalized_entropy(cats, "a"))  # 1.0: two values, equally common
```

## Profiling features before modeling

{py:meth}`Dataset.profile <batcher.Dataset.profile>` answers the data-quality question of how much is present and how many distinct
values there are. `feature_profile` answers the modeling one in the same single pass, and names the
transform each column is asking for:

```python
from batcher.ml.selection import feature_profile

ds = bt.from_pydict(
    {
        "flat": [1.0] * 8,
        "skewed": [float(2**i) for i in range(8)],
        "ok": [float(i) for i in range(8)],
    }
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

`f_classif_scores` and `f_regression_scores` score every feature in **one pass** over the
data, however wide the table is. Both statistics are recoverable from mergeable moments, so
a hundred features ride a single `group_by` rather than costing a hundred scans. That is
what makes them usable as a first screen on a wide frame, and on the distributed path where
each scan is a pass across the cluster.

`chi2_scores` and `mutual_info_scores` still cost a pass per feature: each needs its own
contingency grid over a different pair of category sets, and those do not share a scan.
Screen with the F scores first if the table is wide.

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
is model-agnostic. That makes it honest in a way a tree's built-in importance is not: a tree
can call a feature important because it split on it, even when permuting the feature changes
nothing.

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
flagged = flag_outliers(ds, "latency", method="iqr")  # a boolean flag column, not a drop
```

`flag_outliers` marks them (the decision to keep or drop is yours), `count_outliers` tallies
them, and {py:class}`OutlierClipper <batcher.ml.outliers.OutlierClipper>` clamps them as a fitted preprocessor that applies the *training*
bounds to serving data. `outlier_bounds` returns the raw cut points.

Those rules are univariate, judging one column at a time. `mahalanobis_distance` is the multivariate score for a row that looks ordinary on every column but is an outlier in the *joint* distribution, measuring distance from the center in units that account for the columns' correlations. Its square is chi-squared with one degree of freedom per column, which is how you turn it into a threshold.

```python
from batcher.ml.outliers import mahalanobis_distance

ds = bt.from_pydict({"height": [60.0, 65.0, 70.0, 62.0], "weight": [120.0, 150.0, 180.0, 40.0]})
scored = mahalanobis_distance(ds, ["height", "weight"])
print(
    scored.to_pydict()["mahalanobis"][3] == max(scored.to_pydict()["mahalanobis"])
)  # the light-but-average-height row
```

`mahalanobis_distance` relearns the centre and the covariance from whatever dataset you hand it. That is what you want for a one-off audit and the wrong thing for scoring new data: a batch made entirely of outliers relearns itself as normal and comes back clean. {py:class}`EllipticEnvelope <batcher.ml.outliers.EllipticEnvelope>` splits the two steps, so the envelope is learned once on the training data and applied unchanged to whatever arrives:

```python
from batcher.ml import EllipticEnvelope

train = bt.from_pydict(
    {
        "height": [1.70, 1.75, 1.80, 1.85, 1.65, 1.78, 1.72, 1.82],
        "weight": [64.0, 70.0, 76.0, 82.0, 58.0, 73.0, 66.0, 79.0],
    }
)

envelope = EllipticEnvelope(["height", "weight"], contamination=0.05).fit(train)

incoming = bt.from_pydict({"height": [1.76, 1.79], "weight": [71.0, 55.0]})
print(envelope.predict(incoming).to_pydict()["is_outlier"])
# [False, True]
```

The second row is the case the univariate rules miss: 1.79m and 55kg are each unremarkable, and the pair is not. `contamination` is the share of training rows expected to fall outside, which sets the chi-squared cutoff, and `score_samples` returns the distance itself when you would rather rank rows than threshold them.

The fit is a mean and a covariance, both mergeable aggregates, so it is one pass and gives the same envelope on a cluster as on one machine. Scoring adds no pass at all, because it lowers to a single expression.

## Drift monitoring

When a model has been deployed and the labels have not arrived yet, the only observable thing is whether the *inputs* still look like the training data.

```python
from batcher.ml.stats import drift_report, population_stability_index

train = bt.from_pydict({"x": [float(i) for i in range(200)]})
today = bt.from_pydict({"x": [float(i) + 60 for i in range(200)]})
print(round(population_stability_index(train, today, "x", buckets=5), 4))
```

The bin edges always come from the *reference* distribution, then apply unchanged to the current data. A shift shows up as mass moving between bins rather than as the bins themselves moving. Deriving edges separately for each side would make two very different distributions look identical.

The figure runs the example above through those steps. Five quantile bins of the reference hold 20% each. The same edges put 0%, 10%, 20%, 20%, and 50% of today's shifted data in those bins, and the PSI over those shares is far past the significant band.

![Three panels for the example above. First, the reference column x from 0 to 199 is cut at its quantiles into five bins with edges at 39.8, 79.6, 119.4, and 159.2, so each bin holds 20% of the rows and the outer bins are open-ended. Second, today's column, the same values shifted by 60, is binned on those same reference edges and lands 0%, 10%, 20%, 20%, and 50% in them, so the mass moves between bins while the bins stay put. Third, the PSI sums (current minus reference) times ln(current over reference) over the five bins, with an empty bin counted as 1e-6, giving 2.7854, which reads as significant: below 0.1 is stable, 0.1 to 0.25 moderate, above 0.25 significant.](/_static/diagrams/drift_psi_bins.svg)

The following table gives the usual reading of each measure. The PSI and information-value bands are rules of thumb, not tests:

| Measure | Reading |
|---|---|
| `population_stability_index` | Below 0.1 stable; 0.1 to 0.25 moderate; above 0.25 significant. |
| `js_divergence` | 0 identical, 1 bit maximally different. Comparable across columns. |
| `kl_divergence` | Asymmetric: punishes current mass where the reference had almost none. |
| `categorical_drift` | The share of mass that would have to move for the two to match. |
| `information_value` | Below 0.02 useless; 0.02 to 0.1 weak; 0.1 to 0.3 medium; above 0.3 strong. |

`drift_report` runs the whole check and returns a `Dataset` ordered by descending PSI, so you can append it to a monitoring table. A single PSI says far less than its history:

```python
report = drift_report(train, today, ["x"], buckets=5)
print(report.columns)
```

`woe_table` and `information_value` are the scorecard pair: bin a feature and report the log odds of a positive in each bin. A monotone WOE column is what makes a feature usable in a linear scorecard, and the shape of the table tells you where to merge bins.

## Hypothesis tests

A test statistic says how large an effect is. The p-value says how surprising it is under the null hypothesis, and that is the number you act on. `batcher.ml.stats` pairs each statistic with its p-value in one pass and returns a {py:class}`TestResult <batcher.ml.stats.TestResult>` carrying the statistic, its degrees of freedom, and the p-value.

The following table lists the parametric tests by the question each answers:

| Test | Question |
|---|---|
| `t_test_1samp` | Does a column's mean differ from a target? |
| `t_test_ind` | Do two groups' means differ? Welch's test, so unequal variances are fine. |
| `anova_test` | Do several groups' means differ? |
| `chi_square_test` | Are two categorical columns independent? |
| `normality_test` | Is a column plausibly Gaussian? Jarque-Bera, as a screen. |
| `pearson_test`, `spearman_test` | Is a linear or monotone correlation real? |
| `proportion_ztest`, `binomial_test` | Does a success rate differ from a target? `binomial_test` is exact for small samples. |
| `mcnemar_test` | Does one classifier make fewer errors than another on the same rows? |

`mcnemar_test` is the paired test to reach for when deciding whether one model genuinely beats another.

```python
import batcher as bt
from batcher.ml.stats import t_test_ind, anova_test

ds = bt.from_pydict({"g": ["a", "a", "a", "b", "b", "b"], "x": [1.0, 2.0, 3.0, 8.0, 9.0, 10.0]})
result = t_test_ind(ds, "x", "g")
print(round(result.pvalue, 4), result.pvalue < 0.05)
```

`bartlett_test` and `levene_test` check the equal-variance assumption a t-test and an ANOVA quietly rely on. Bartlett's has more power on normal groups. Levene's, median-centered, is the robust default.

When the data itself is too skewed or ordinal for a t-test, `mann_whitney_u` (two groups) and `kruskal_wallis` (several) are the rank-based, distribution-free alternatives, asking whether one group tends to larger ranks rather than a larger mean. For *paired* measurements such as a before/after or matched-pair design, `wilcoxon_signed_rank` is the distribution-free replacement for the paired t-test. `friedman_test` extends that to several treatments measured on the same blocks: the non-parametric repeated-measures ANOVA.

Report `cliffs_delta` or `common_language_effect_size` beside a Mann-Whitney result. The test says *whether* two groups differ; these say *how much*, as the probability that a random member of one exceeds a random member of the other.

The tail probabilities come from dependency-free implementations of the Student's t, F, and chi-squared survival functions, which the test suite checks against SciPy. The reduction is a handful of aggregates, so a test scales like every other statistic here.

### Missing values and degenerate input

Every function in `batcher.ml.stats` follows SciPy's defaults on input that is not a clean sample, so a result can be checked against `scipy.stats` directly:

- A null is a missing observation and is dropped. That holds in the group column too, where a row with a null label is left out rather than forming a group of its own.
- A NaN is a value, and it propagates. The statistic and the p-value are NaN, which is SciPy's `nan_policy="propagate"`. A NaN never reads as `p = 0`.
- Too little data gives NaN rather than an exception. That covers a group of one row, all-constant data, a paired test whose differences are all zero, and an empty dataset. Constant groups that differ from each other give an infinite statistic with `p = 0`, as SciPy reports.
- Too few groups for the question raises {py:exc}`PlanError <batcher.PlanError>`, as SciPy raises for it. A between-groups test needs at least two groups after nulls are dropped, and `t_test_ind` and `mann_whitney_u` need exactly two.

```python
import math

from batcher.ml.stats import kruskal_wallis, t_test_1samp

messy = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 10.0], "g": [*"aaabbbbb", None]})
print(kruskal_wallis(messy, "x", "g").statistic)  # 5.0: the null-label row is not a group

with_nan = bt.from_pydict({"x": [1.0, 2.0, float("nan"), 4.0]})
assert math.isnan(t_test_1samp(with_nan, 0.0, "x").pvalue)
```

`variance_inflation_factor` follows the same rule: with no more complete rows than columns the regression behind each VIF is underdetermined, and every value is NaN. A column that is an exact linear combination of the others has an infinite VIF.

### On a cluster

None of these functions takes a `distributed=` argument. Each runs its aggregates through `collect()` with the default `distributed="auto"`, which decides from the input size and the cluster. To force them onto Ray, or keep them off it, set the session pin:

```python
# docs: skip
from batcher.config import option_context

with option_context("distributed.mode", "always"):
    result = kruskal_wallis(big_table, "latency", "region")
```

The answer is the same either way, up to float reassociation in the last bits.

## Time-series diagnostics

A time series carries its signal in how a column relates to its own past, which a Pearson correlation can't see. `batcher.ml.timeseries` orders a column by a time key and measures that self-relationship. `autocorrelation` gives the lag-`k` value and `autocorrelations` the whole function up to a maximum lag. `ljung_box` pools the first several lags into one white-noise test, and `durbin_watson` is the regression diagnostic for autocorrelated residuals.

`partial_autocorrelation` and `partial_autocorrelations` give the *partial* function, which strips out what the intervening lags already explain. It cuts off sharply at the order of an autoregressive process, and that cutoff is how you choose the order.

```python
import batcher as bt
from batcher.ml.timeseries import autocorrelations, ljung_box

ds = bt.from_pydict({"t": list(range(12)), "sales": [float(i % 4) for i in range(12)]})
print({k: round(v, 3) for k, v in autocorrelations(ds, "sales", 4, order_by="t").items()})
print(ljung_box(ds, "sales", 4, order_by="t").pvalue < 0.05)
```

For scoring a forecast, `mean_absolute_scaled_error` is the scale-free metric: the model's mean absolute error divided by the naive seasonal forecast's, so a value below 1 beats naive and the number is comparable across series on any scale.

An autocorrelation needs the whole series in time order, so unlike the mergeable statistics above these run over a single ordered window rather than a partitionable aggregate. The formulas are the Box-Jenkins definitions, and the test suite pins each to an independent numpy computation.

## Requirements and limitations

A drift measure needs a reference column with more than one distinct value. A constant reference raises rather than reporting 0.0, because "no drift" for a column that moved from 1.0 to 2.0 is the worst possible answer.

`js_divergence` does not reach 1 for a wholly shifted column, because the outermost reference bins are open-ended and absorb everything beyond them. Alert on `population_stability_index`, which has no such ceiling; use JS to compare across columns.

## See also

- {doc}`/ml/evaluation/splits-and-resampling`: rebalance a rare class, hold out a test set, and build the folds.
- {doc}`/ml/evaluation/evaluation`: score a model once you have a trustworthy split.
- {doc}`/ml/preparing/preprocessors/feature-selection`: act on these screens inside a fitted pipeline.
- {doc}`/ml/preparing/preprocessors/index`: the transforms these statistics tell you a column needs.
- {doc}`/user-guide/trust/data-quality`: assert contracts rather than measure them.
- {doc}`/cookbook/metrics/statistics/index`: short runnable recipes for the functions on this page.
