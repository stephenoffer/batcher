# Evaluate a model

This page describes how to score predictions against labels in Batcher, and how to get the same metrics per segment without a second pass over the data.

## Why metrics are expressions here

Every metric on this page is an `Expr`, not a function that takes two arrays. That one decision is what makes the surface different from `sklearn.metrics`:

- The metrics run **in the engine**, so evaluating a billion scored rows never materializes them on a driver.
- Asking for ten metrics costs what asking for one costs, because they reduce to the same aggregate pass.
- They compose with `group_by`, so "what is the F1 *per country, per month*" is the same query with a grouping added — the question a model review actually asks, and the one a driver-side call cannot answer at scale.

The exceptions are the metrics that need a global ordering rather than a per-row quantity: ROC AUC, average precision, and the KS statistic. Those are Dataset functions built on a window rank, and each adds one sort.

## One call for a whole task

`ds.ml.evaluate` runs a task's default metric set:

```python
import batcher as bt

ds = bt.from_pydict(
    {
        "y": [1, 0, 1, 0, 1, 0],
        "score": [0.9, 0.1, 0.8, 0.4, 0.7, 0.2],
    }
)
report = ds.ml.evaluate("y", y_score="score")
print(report["accuracy"], round(report["roc_auc"], 4))
```

Giving `y_score` alone is enough for a binary task: the hard predictions are derived at `threshold` (0.5 by default), so the threshold metrics and the ranking metrics both come from one scored column. Pass `y_pred` instead when you already have hard labels, and `task="regression"` when the target is continuous.

`metrics=` narrows the set, which matters when a rank metric's sort is not worth paying for:

```python
print(ds.ml.evaluate("y", y_score="score", metrics=["precision", "recall", "f1"]))
```

## The same metrics per segment

Add `by=` and the result is a `Dataset` with one row per group rather than a dict:

```python
ds = bt.from_pydict(
    {
        "region": ["eu", "eu", "us", "us"],
        "y": [1, 0, 1, 0],
        "score": [0.9, 0.2, 0.3, 0.7],
    }
)
per_region = ds.ml.evaluate("y", y_score="score", by="region", metrics=["accuracy", "roc_auc"])
print(per_region.sort("region").to_pydict())
```

This is the query worth reaching for first when a model looks fine overall. An aggregate accuracy of 0.94 routinely hides a segment at 0.61, and only the grouped form shows it.

## Individual metrics inside any aggregate

Because the metrics are expressions, they go anywhere an aggregate goes:

```python
ds = bt.from_pydict({"y": [1.0, 2.0, 3.0, 4.0], "p": [1.1, 2.2, 2.7, 4.4]})
print(ds.agg(rmse=bt.rmse("y", "p"), mae=bt.mae("y", "p"), r2=bt.r2("y", "p")).to_pydict())
```

The full vocabulary:

| Family | Metrics |
|---|---|
| Regression error | `mse`, `rmse`, `normalized_rmse`, `mae`, `medae`, `max_error`, `mean_bias`, `mean_percentage_error` |
| Relative error | `mape`, `smape`, `wape`, `msle`, `rmsle` |
| Fit | `r2`, `explained_variance` |
| Robust objectives | `huber_loss`, `pinball_loss` |
| Confusion counts | `true_positives`, `false_positives`, `false_negatives`, `true_negatives` |
| Rates | `accuracy`, `precision`, `recall`, `specificity`, `false_positive_rate`, `false_negative_rate`, `negative_predictive_value`, `prevalence` |
| Balanced | `f1_score`, `fbeta_score`, `balanced_accuracy`, `matthews_corrcoef`, `cohen_kappa` |
| Calibration | `log_loss`, `brier_score` |
| Multi-label | `hamming_loss` (the fraction of label cells predicted wrong) |
| Diagnostic-test | `jaccard_score`, `false_discovery_rate`, `false_omission_rate`, `positive_likelihood_ratio`, `negative_likelihood_ratio`, `diagnostic_odds_ratio`, `informedness`, `markedness`, `fowlkes_mallows_index`, `geometric_mean_score`, `prevalence_threshold` |

Every one of them is checked against `sklearn.metrics` at 1e-12 in the test suite, so the definitions are the ones you expect.

## Choosing the right metric

A few of these exist specifically because the obvious choice misleads, and it is worth knowing which:

`accuracy` is misleading on imbalanced data. At 1% positives, predicting "negative" always scores 0.99. Report `balanced_accuracy` or `matthews_corrcoef` beside it.

`mape` is undefined where the actual is zero, and Batcher excludes those rows from *both* the numerator and the denominator. Use `wape` when zeros are common: it is a ratio of totals rather than a mean of ratios, so one near-zero actual cannot dominate it.

`mean_percentage_error` keeps the sign `mape` discards, so it measures a forecast's *bias* — a positive value means it systematically under-predicts. `normalized_rmse` divides the RMSE by the mean of the actuals, making it comparable across series on different scales. On the classification side `false_negative_rate` (`1 - recall`) is the miss rate to watch when an undetected positive is the costly outcome.

`roc_auc` counts every negative equally, so at very low prevalence it stays high while the top of the ranking is worthless. Report `average_precision` instead when positives are rare.

`log_loss` and `brier_score` are the only metrics here that score *calibration*. A model that ranks perfectly but predicts probabilities twice as large as the truth looks excellent on AUC and fails the moment a prediction is multiplied by a dollar amount.

`hinge_loss` and `squared_hinge_loss` score a raw decision function rather than a probability — the margin a support-vector machine or a linear classifier produces. They are zero once a point is correctly classified with room to spare and grow with how far a point sits on the wrong side, which is the objective those models optimize.

## Diagnostic tables

A single metric says how good a model is; these say *where* it is wrong. Each returns a lazy `Dataset`, so the result joins, filters, and writes:

```python
from batcher.ml.metrics import calibration_curve, confusion_matrix, lift_table

ds = bt.from_pydict({"y": [1, 1, 0, 0], "p": [1, 0, 0, 0], "s": [0.9, 0.8, 0.2, 0.1]})
print(confusion_matrix(ds, "y", "p").sort("y", "p").to_pydict())
print(lift_table(ds, "y", "s", buckets=2).to_pydict()["lift"])
print(calibration_curve(ds, "y", "s", bins=2).to_pydict()["observed_rate"])
```

`confusion_matrix` is long form — one row per `(actual, predicted)` pair — which is what stays correct when the label set is large or discovered from the data, and what joins to a cost table.

`threshold_sweep` reports precision, recall, and the four confusion counts at every cutoff in one pass, which is what picking an operating point actually needs.

`lift_table` cuts the ranking into equal-sized buckets and reports each one's positive rate against the base rate. It is the table a marketing or risk team reads.

`calibration_curve` bins the predicted probability and reports the observed frequency beside it. A well-calibrated model has the two equal in every row.

## More than two classes

A single accuracy across ten classes routinely hides one the model never predicts at all,
and that class is usually the one anybody cared about. `classification_report` is the table
to read instead, and every class's counts come from the *same* aggregate pass:

```python
from batcher.ml.metrics import classification_report

ds = bt.from_pydict({"y": ["a", "a", "b", "c"], "p": ["a", "b", "b", "c"]})
print(classification_report(ds, "y", "p").sort("class").to_pydict()["f1"])
```

`evaluate(task="multiclass")` adds the two ways of averaging those per-class numbers, and
reporting only one of them is how a model that ignores the minority class passes review.
The **macro** average weights every class equally, so a rare class the model ignores drags
it down; the **weighted** average weights by support, so it tracks accuracy and a rare class
barely moves it.

```python
print(round(ds.ml.evaluate("y", y_pred="p", task="multiclass")["macro_f1"], 4))
```

## Choosing an operating point

A classifier outputs a score, not a decision. Turning it into an action needs a cutoff, and
0.5 is right only when the classes are balanced *and* a false positive costs exactly what a
false negative costs — which is close to never true.

`best_threshold` finds the cutoff that maximizes a metric, in one pass plus an argmax over
the candidates:

```python
from batcher.ml.metrics import best_threshold

ds = bt.from_pydict({"y": [0, 0, 1, 1], "s": [0.1, 0.2, 0.8, 0.9]})
print(round(best_threshold(ds, "y", "s", thresholds=10)["f1"], 4))
```

Better, when you can name the costs: `best_cost_threshold` minimizes expected cost directly
rather than a proxy for it. F1 implicitly assumes a miss and a false alarm cost the same, and
at a 10:1 ratio the F1-optimal cutoff can cost more than twice the minimum.

```python
from batcher.ml.metrics import best_cost_threshold

best = best_cost_threshold(ds, "y", "s", cost_false_positive=1.0, cost_false_negative=10.0)
print(best["threshold"], best["total_cost"])
```

`expected_cost_curve` returns the whole curve when you want to see the shape rather than
just the minimum — a flat curve means the cutoff barely matters, which is worth knowing
before anyone argues about it.

## Comparing candidate models

Because the metrics are aggregate expressions, several models' metrics go into the *same*
`agg` — so comparing six candidates costs one scan, not six evaluations:

```python
from batcher.ml.metrics import compare_models

ds = bt.from_pydict(
    {"y": [1, 0, 1, 0], "model_a": [0.9, 0.1, 0.8, 0.2], "model_b": [0.4, 0.6, 0.4, 0.6]}
)
table = compare_models(ds, "y", {"a": "model_a", "b": "model_b"}, metrics=["accuracy", "f1"])
print(table.sort("f1", descending=True).to_pydict()["model"])
```

The result is a `Dataset`, so it sorts, joins to a latency or serving-cost column, and
appends to an experiment log — which is what turns a comparison into a record of why a model
was chosen. Rank-based metrics need a sort each and are refused rather than silently made
slow; ask for those per model with `roc_auc(..., by=)`.

## Fairness

A model can be accurate overall and systematically worse for one group, and no aggregate
metric shows it. The fairness metrics compare the model's behaviour across a protected
attribute and report the gap — and because the definitions disagree (you cannot satisfy all
at once), naming them separately forces the choice to be explicit.

```python
from batcher.ml.metrics import demographic_parity_difference, equal_opportunity_difference

ds = bt.from_pydict(
    {"race": ["a", "a", "b", "b"], "y": [1, 0, 1, 0], "p": [1, 1, 0, 0]}
)
print(demographic_parity_difference(ds, "race", "p"))       # selection-rate gap
print(equal_opportunity_difference(ds, "race", "y", "p"))   # true-positive-rate gap
```

`demographic_parity_difference` and `disparate_impact_ratio` measure equal *selection*;
`equal_opportunity_difference` and `equalized_odds_difference` measure equal *error rates*;
`predictive_parity_difference` measures whether a positive prediction means the same thing in
each group. `group_fairness_report` returns the per-group rates the disparities are computed
from. None of these is a verdict — a gap can be justified — but an unmeasured gap cannot.

## Agreement, not just correlation

A correlation says two series move together; it says nothing about whether they are *equal*. A
prediction that is always half the truth correlates perfectly and is useless. `bt.concordance_correlation`
(Lin's CCC) penalises a correlation by how far the means and variances differ;
`bt.nash_sutcliffe_efficiency` and `bt.kling_gupta_efficiency` are the hydrology efficiency
scores that decompose agreement into correlation, bias, and variability.

```python
ds = bt.from_pydict({"y": [1.0, 2.0, 3.0, 4.0], "p": [3.0, 4.0, 5.0, 6.0]})
print(round(ds.agg(m=bt.concordance_correlation("y", "p")).to_pydict()["m"][0], 4))  # < 1: shifted
```

## Rank-based metrics

```python
from batcher.ml.metrics import average_precision, gini_coefficient, ks_statistic, roc_auc

ds = bt.from_pydict({"y": [0, 0, 1, 1], "s": [0.1, 0.4, 0.35, 0.8]})
print(roc_auc(ds, "y", "s"), round(average_precision(ds, "y", "s"), 4))
print(ks_statistic(ds, "y", "s"), gini_coefficient(ds, "y", "s"))
```

ROC AUC uses the rank identity rather than integrating a threshold sweep, so it is exact including under ties and needs one sort rather than one scan per threshold. Each of these takes `by=` for a per-segment value.

## Is the probability calibrated?

A model can rank perfectly and still lie about its confidence — an excellent AUC with a
useless probability. Calibration is the property AUC cannot see, and it is what matters the
moment a predicted probability is multiplied by a dollar amount.

```python
from batcher.ml.metrics import brier_skill_score, expected_calibration_error

ds = bt.from_pydict({"y": [0, 0, 1, 1, 1], "s": [0.1, 0.3, 0.6, 0.8, 0.9]})
print(round(expected_calibration_error(ds, "y", "s", bins=5), 4))
print(round(brier_skill_score(ds, "y", "s"), 4))
```

`expected_calibration_error` is the support-weighted average gap between predicted confidence
and observed accuracy; `maximum_calibration_error` is the worst band's gap, for when one
region of the score range drives a high-stakes call. `brier_skill_score` rescales the Brier
score against the base rate so it reads like R²: 1 is perfect, 0 is no better than predicting
the base rate, negative is worse.

## Count and rate models

RMSE assumes symmetric, constant-variance error. A count (claims, clicks, defects) or a rate
is neither, and a Poisson, gamma, or Tweedie model is fitted on the matching *deviance*
instead — so the deviance is the honest way to score it. All three are expressions, checked
against scikit-learn:

```python
ds = bt.from_pydict({"y": [0.0, 2.0, 5.0], "p": [1.0, 2.0, 4.0]})
print(round(ds.agg(m=bt.poisson_deviance("y", "p")).to_pydict()["m"][0], 4))
```

`bt.gamma_deviance` handles a positive, right-skewed target (a claim size, a duration). `bt.tweedie_deviance(y, p, power=...)` spans the whole family — 1 is Poisson, 2 is gamma, and a
power in ``(1, 2)`` is the compound distribution that describes insurance pure premium.
`d2_tweedie_score` (in `batcher.ml.metrics`) is the deviance-explained score, R² for a count
model. `d2_absolute_error_score` and `d2_pinball_score` are the same idea on the L1 and quantile
scales: each reports the fraction of its own loss a model explains against the optimal constant
baseline (the target's median for absolute error, its `alpha`-quantile for pinball), so a
median or quantile regression gets the same self-contained 0-to-1 score R² gives a
least-squares one.

## Regression diagnostics

A single RMSE hides where the error lives. `residual_summary` groups the residual — its
mean (systematic bias), spread, and quantiles — by any column, which is what surfaces a
model that is unbiased overall while badly over-predicting one segment:

```python
from batcher.ml.metrics import residual_summary

ds = bt.from_pydict({"seg": ["a", "a", "b"], "y": [10.0, 10.0, 10.0], "p": [12.0, 12.0, 8.0]})
print(residual_summary(ds, "y", "p", by="seg").sort("seg").to_pydict()["mean_residual"])
```

`prediction_interval_coverage` checks a quantile model's central promise: that a "90%
interval" actually contains ≈90% of the actuals. `top_k_accuracy` scores a ranked
multi-class prediction — was the true label among the model's top `k` guesses — which is the
honest number for a recommender where the top-1 label being wrong is not a failure.

## Baselines

A score means nothing without a floor to read it against. `batcher.ml.dummy.DummyRegressor` (predict the target's mean or median) and `DummyClassifier` (predict the majority class) are that floor: fit and score them on the same split, and a real model has to clear them to have earned its complexity. On an imbalanced target the dummy classifier is the number a high accuracy is quietly matching.

```python
import batcher as bt
from batcher.ml.dummy import DummyClassifier

ds = bt.from_pydict({"y": [0, 0, 0, 0, 1]})
print(DummyClassifier("y").fit(ds).constant_)  # the majority class a model must beat
```

## Requirements and limitations

`average_precision` breaks ties in the engine's sort order, so a score column with heavy ties gives an optimistic value. `roc_auc` is exact under ties and is the safer choice there.

A ranking metric on a split containing only one class is undefined and returns NaN. Check the class balance of a segment before trusting a per-segment AUC.

The multi-class averages are computed from a per-class report rather than a single aggregate, so `by=` cannot partition them. Group the dataset and call `evaluate` per group when you need both.

## See also

- {doc}`tabular-models` — produce the predictions this page scores.
- {doc}`statistics-and-drift` — watch the inputs when the labels have not arrived yet.
- {doc}`../user-guide/aggregations` — the aggregate surface these metrics are built on.
