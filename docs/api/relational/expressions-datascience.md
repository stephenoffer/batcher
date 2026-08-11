# Data science and evaluation expressions

This page tabulates the feature-engineering, profiling and model-evaluation expressions.
The core column language they compose with is in {doc}`/api/relational/expressions`.

```python
import batcher as bt
```

## Data science toolkit

Feature engineering, profiling, and text/calendar features as expressions, so a
fit-and-apply transform is one pass over Arrow with no Python state. They're ordinary window and arithmetic nodes, so they're identical single-node and distributed.

**Scaling and encoding** (each takes `partition_by=` to fit per group):
{py:meth}`.zscore() <batcher.plan.expr_ir.core.Expr.zscore>` (standardize), {py:meth}`.minmax_scale() <batcher.plan.expr_ir.core.Expr.minmax_scale>`, {py:meth}`.maxabs_scale() <batcher.plan.expr_ir.core.Expr.maxabs_scale>`, {py:meth}`.mean_center() <batcher.plan.expr_ir.core.Expr.mean_center>`,
{py:meth}`.label_encode() <batcher.plan.expr_ir.core.Expr.label_encode>` (0-based codes by sorted value), and {py:meth}`.hash_bucket(n, seed=0) <batcher.plan.expr_ir.core.Expr.hash_bucket>` for
reproducible shard / split assignment.

**Activations and shape**: {py:meth}`.sigmoid() <batcher.plan.expr_ir.core.Expr.sigmoid>`, {py:meth}`.logit() <batcher.plan.expr_ir.core.Expr.logit>`, {py:meth}`.relu() <batcher.plan.expr_ir.core.Expr.relu>`, {py:meth}`.softplus() <batcher.plan.expr_ir.core.Expr.softplus>`, {py:meth}`.silu() <batcher.plan.expr_ir.core.Expr.silu>`
(Swish, `x·sigmoid(x)`), {py:meth}`.gelu() <batcher.plan.expr_ir.core.Expr.gelu>` (the transformer default, tanh approximation), {py:meth}`.mish() <batcher.plan.expr_ir.core.Expr.mish>`,
{py:meth}`.hardsigmoid() <batcher.plan.expr_ir.core.Expr.hardsigmoid>` / {py:meth}`.hardswish() <batcher.plan.expr_ir.core.Expr.hardswish>` (the cheap piecewise-linear MobileNet variants),
{py:meth}`.leaky_relu(negative_slope=0.01) <batcher.plan.expr_ir.core.Expr.leaky_relu>`, {py:meth}`.elu(alpha=1.0) <batcher.plan.expr_ir.core.Expr.elu>`, {py:meth}`.hardtanh() <batcher.plan.expr_ir.core.Expr.hardtanh>`, {py:meth}`.softsign() <batcher.plan.expr_ir.core.Expr.softsign>`,
{py:meth}`.tanhshrink() <batcher.plan.expr_ir.core.Expr.tanhshrink>`, and
`.softmax()` (scores to a distribution summing to 1). Each matches its `torch.nn.functional`
counterpart and runs in the data plane.

**Comparison and de-duplication**: {py:meth}`.abs_diff(other) <batcher.plan.expr_ir.core.Expr.abs_diff>`, plus
{py:meth}`.is_first_distinct(order_by) <batcher.plan.expr_ir.core.Expr.is_first_distinct>` / {py:meth}`.is_last_distinct(order_by) <batcher.plan.expr_ir.core.Expr.is_last_distinct>`, which mark one row per
distinct value (the `order_by` is required so the pick is partition-independent).

**Ratios and shares**: {py:meth}`.pct_of_total() <batcher.plan.expr_ir.core.Expr.pct_of_total>`, {py:meth}`.cumulative_pct() <batcher.plan.expr_ir.core.Expr.cumulative_pct>` (the Pareto curve),
{py:meth}`.normalize_l1() <batcher.plan.expr_ir.core.Expr.normalize_l1>`, {py:meth}`.rank_pct() <batcher.plan.expr_ir.core.Expr.rank_pct>` (percentile rank), and {py:meth}`.safe_divide(other) <batcher.plan.expr_ir.core.Expr.safe_divide>`, which
yields null rather than infinity when the divisor is zero.

**Expanding (cumulative) statistics**: {py:meth}`.expanding_mean() <batcher.plan.expr_ir.core.Expr.expanding_mean>`, {py:meth}`.expanding_var() <batcher.plan.expr_ir.core.Expr.expanding_var>`,
{py:meth}`.expanding_std() <batcher.plan.expr_ir.core.Expr.expanding_std>`, the growing-frame counterparts of the `rolling_*` family.

**Value predicates**: {py:meth}`.is_positive() <batcher.plan.expr_ir.core.Expr.is_positive>`, {py:meth}`.is_negative() <batcher.plan.expr_ir.core.Expr.is_negative>`, {py:meth}`.is_zero() <batcher.plan.expr_ir.core.Expr.is_zero>`, {py:meth}`.is_even() <batcher.plan.expr_ir.core.Expr.is_even>`,
{py:meth}`.is_odd() <batcher.plan.expr_ir.core.Expr.is_odd>`, and {py:meth}`.is_outlier(threshold=3.0) <batcher.plan.expr_ir.core.Expr.is_outlier>` (the z-score rule, as a filterable
predicate).

**Calendar features** on `.dt`: {py:meth}`.is_weekend() <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.is_weekend>` / {py:meth}`.is_weekday() <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.is_weekday>`,
{py:meth}`.is_month_start() <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.is_month_start>` / {py:meth}`.is_month_end() <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.is_month_end>`, {py:meth}`.is_quarter_start() <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.is_quarter_start>` / {py:meth}`.is_quarter_end() <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.is_quarter_end>`,
{py:meth}`.is_year_start() <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.is_year_start>` / {py:meth}`.is_year_end() <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.is_year_end>`, {py:meth}`.quarter_start() <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.quarter_start>`, {py:meth}`.year_start() <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.year_start>`,
{py:meth}`.days_in_year() <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.days_in_year>`, and {py:meth}`.week_of_month() <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.week_of_month>`.

**Time deltas** on `.dt`: {py:meth}`.seconds_between(other) <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.seconds_between>`, {py:meth}`.minutes_between(other) <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.minutes_between>`,
{py:meth}`.hours_between(other) <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.hours_between>`, {py:meth}`.days_between(other) <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.days_between>`, and {py:meth}`.weeks_between(other) <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.weeks_between>` measure
elapsed fixed-width time between two timestamps; {py:meth}`.quarter_end() <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.quarter_end>` and {py:meth}`.year_end() <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.year_end>`
complete the period boundaries.

**Text features** on `.str`: {py:meth}`.word_count() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.word_count>`, {py:meth}`.digit_count() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.digit_count>`, {py:meth}`.contains_all([...]) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.contains_all>`,
{py:meth}`.count_char(sub) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.count_char>`, {py:meth}`.is_alpha() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_alpha>`,
{py:meth}`.is_numeric() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_numeric>`, {py:meth}`.is_alnum() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_alnum>`, {py:meth}`.is_space() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_space>`, {py:meth}`.is_upper() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_upper>`, {py:meth}`.is_lower() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_lower>`,
{py:meth}`.capitalize() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.capitalize>`, and {py:meth}`.remove_punctuation() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.remove_punctuation>`.

```python
feats = bt.from_pydict({"g": ["a", "a", "b", "b"], "v": [1.0, 3.0, 10.0, 20.0]})
out = feats.select(
    z=bt.col("v").zscore(["g"]).round(4),
    share=bt.col("v").pct_of_total(["g"]),
    bucket=bt.col("g").hash_bucket(2),
)
print(out.to_pydict())
# {'z': [-0.7071, 0.7071, -0.7071, 0.7071], 'share': [0.25, 0.75, 0.3333333333333333, 0.6666666666666666], 'bucket': [1, 1, 1, 1]}
```

**Weighted statistics** (when rows carry survey, recency, or size weights):
{py:func}`bt.weighted_mean(x, w) <batcher.weighted_mean>`, {py:func}`bt.weighted_var(x, w) <batcher.weighted_var>`, {py:func}`bt.weighted_std(x, w) <batcher.weighted_std>`,
{py:func}`bt.weighted_covariance(x, y, w) <batcher.weighted_covariance>`, and {py:func}`bt.weighted_correlation(x, y, w) <batcher.weighted_correlation>` are each the
frequency-weighted form matching `numpy.average`.

Column-level profiling aggregates complete the toolkit: {py:func}`bt.q1(x) <batcher.q1>` / {py:func}`bt.q3(x) <batcher.q3>` /
{py:func}`bt.iqr(x) <batcher.iqr>` (robust spread), {py:func}`bt.value_range(x) <batcher.value_range>`, {py:func}`bt.null_rate(x) <batcher.null_rate>` /
{py:func}`bt.non_null_rate(x) <batcher.non_null_rate>` (completeness), and {py:func}`bt.nunique_ratio(x) <batcher.nunique_ratio>`, the cardinality ratio, where near 1 marks an identifier and near 0 a categorical.

## Model evaluation metrics

Every model-evaluation metric is an expression, so it belongs inside `agg()` and composes
with `group_by`. A per-segment report is the same query with a grouping added, at no extra
pass. All are checked against scikit-learn where it defines them.

They are top-level functions rather than `Expr` methods, so they are enumerated on
{doc}`/api/models/metrics` with their signatures and docstrings.

```python
scored = bt.from_pydict({"y": [1, 0, 1, 1, 0], "p": [1, 0, 0, 1, 1]})
print(scored.agg(
    f1=bt.f1_score("y", "p"),
    jaccard=bt.jaccard_score("y", "p"),
    informedness=bt.informedness("y", "p"),
).to_pydict())
```

The metrics that need a global ordering (ROC AUC, average precision) or return a table
(confusion matrix, calibration curve) are Dataset functions in `batcher.ml.metrics`, not
expressions. See {doc}`/ml/evaluation/evaluation`.

## See also

- {doc}`/api/relational/expressions`: operators, null handling, aggregates and window methods.
- {doc}`/api/models/metrics`: the Dataset-level metrics that return a table rather than a value.
- {doc}`/ml/evaluation/evaluation`: choosing and reading these metrics.
