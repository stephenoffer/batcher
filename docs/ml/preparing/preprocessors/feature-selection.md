# Feature selection

This page describes how to cut a wide feature table down to the columns that earn their
place, and how to do it without leaking the validation split into the choice.

Fewer features is not an aesthetic preference. A column that carries no signal still costs
memory, still costs a shuffle when you join, and still gives a model one more chance to fit
noise. Two columns that duplicate each other are worse: a linear fit has to split one
effect between them, so the coefficients come out large, opposite in sign, and unstable
under resampling.

## Selection is fitted state

The single most important thing about a selector is that it is an object. Choose features
on the whole frame, or re-choose them per split, and the held-out rows have participated in
the decision. Your validation score is then optimistic by an amount nothing measures and no
test catches.

```python
import batcher as bt
from batcher.ml.preprocessors import SelectKBest

ds = bt.from_pydict(
    {
        "y": ["a", "a", "a", "b", "b", "b"],
        "signal": [1.0, 1.1, 0.9, 9.0, 9.2, 8.8],
        "weak": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
        "noise": [5.0, 1.0, 5.0, 1.0, 5.0, 1.0],
    }
)
train, test = ds.ml.train_test_split(0.34, seed=0)

selector = SelectKBest("y", k=1).fit(train)
print(selector.selected_)
# ['signal']
print(selector.transform(test).columns)
# ['y', 'signal']
```

`fit` learns the choice, `transform` applies it. The test split is pruned by the *training*
split's decision, whatever the test rows would have scored on their own.

## Choosing a selector

Work down this list, stopping when the table is small enough. Each row costs more than the
one above it:

| Selector | Cost | Sees interaction |
|---|---|---|
| {py:class}`SelectKBest <batcher.ml.preprocessors.SelectKBest>` | One aggregate per feature | No |
| {py:class}`SelectPercentile <batcher.ml.preprocessors.SelectPercentile>` | One aggregate per feature | No |
| {py:class}`DropCorrelated <batcher.ml.preprocessors.DropCorrelated>` | One correlation pass | Pairwise only |
| {py:class}`SelectFromModel <batcher.ml.preprocessors.SelectFromModel>` | One model fit | Yes |
| {py:class}`RFE <batcher.ml.preprocessors.RFE>` | One fit per elimination round | Yes |

## Univariate filtering

{py:class}`SelectKBest <batcher.ml.preprocessors.SelectKBest>` scores every candidate
feature against the target on its own and keeps the best `k`. Each score is a mergeable
aggregate, so scoring a thousand columns is a thousand one-pass reductions rather than a
materialized correlation matrix.

Pick the scorer to match the target and the feature types:

`f_classif`
: Numeric features, categorical target. The default.

`f_regression`
: Numeric features, continuous target.

`chi2`
: Categorical features, categorical target.

`mutual_info`
: Either, and the one that catches a non-monotone relationship the F scores miss.

```python
from batcher.ml.preprocessors import SelectPercentile

regression = bt.from_pydict(
    {
        "y": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "b": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    }
)
print(SelectKBest("y", k=1, score_func="f_regression").fit(regression).selected_)
# ['a']
print(SelectPercentile("y", percentile=50, score_func="f_regression").fit(regression).selected_)
# ['a']
```

Use `SelectPercentile` when the feature count varies between runs. A fixed `k` that suited
a fifty-column table keeps almost nothing from a five-hundred-column one.

A univariate score is blind to interaction by construction. A feature that only matters
alongside another scores as noise, so treat this as a way to remove obvious dead weight
rather than as the final word.

Columns the selector never scored are kept. Only a scored-and-rejected feature is dropped,
so the target, an id column, and anything `features` excluded all survive:

```python
with_id = ds.with_columns(row_id=bt.col("weak"))
kept = SelectKBest("y", k=1, features=["signal", "noise"]).fit_transform(with_id)
print(sorted(kept.columns))
# ['row_id', 'signal', 'weak', 'y']
```

## Removing redundant columns

{py:class}`DropCorrelated <batcher.ml.preprocessors.DropCorrelated>` removes one column
from every pair correlated above a threshold. Which one goes is decided by position, not by
dict ordering, so two runs over the same table produce the same feature set:

```python
from batcher.ml.preprocessors import DropCorrelated

duplicated = bt.from_pydict(
    {"a": [1.0, 2.0, 3.0, 4.0], "a_copy": [2.0, 4.0, 6.0, 8.0], "b": [1.0, 0.0, 1.0, 0.0]}
)
print(DropCorrelated(threshold=0.95).fit(duplicated).dropped_)
# ['a_copy']
```

Pass `keep` for a column that must survive whatever it correlates with. Its partner is
dropped instead, rather than the pair being left standing:

```python
print(DropCorrelated(keep=["a_copy"]).fit(duplicated).dropped_)
# ['a']
```

## Reading a model's own choice

{py:class}`SelectFromModel <batcher.ml.preprocessors.SelectFromModel>` keeps the features a
fitted model gave a large enough coefficient. Paired with a
{py:class}`Lasso <batcher.ml.Lasso>` it is the standard embedded-selection recipe: the L1
penalty drives useless coefficients to exactly zero, and the default `threshold=0` keeps
whatever survived.

```python
from batcher.ml import Lasso
from batcher.ml.preprocessors import SelectFromModel

model = Lasso(["a", "b"], "y", alpha=0.5).fit(regression)
print(SelectFromModel(model).fit(regression).selected_)
# ['a']
```

The estimator must already be fitted. That is deliberate: refitting inside the selector
would hide which rows the selection saw, and that is precisely what decides whether it
leaks.

`threshold` also accepts `"mean"` or `"median"` to cut at that statistic of the
importances, and `max_features` caps the count regardless.

{py:func}`feature_importances <batcher.ml.feature_importances>` is what the selector reads,
and it is worth calling directly when you want to see the magnitudes rather than just the
survivors. It understands Batcher's estimators, scikit-learn's `feature_importances_` and
`coef_`, and a plain dict:

```python
from batcher.ml.preprocessors import feature_importances

print(sorted(feature_importances(model)))
# ['a', 'b']
```

Coefficient magnitudes are only comparable across features when the features are on a
comparable scale, so scale before you fit the model you read.

## Recursive elimination

Dropping a feature changes what the survivors are worth, so ranking once and cutting is not
the same as cutting one at a time. {py:class}`RFE <batcher.ml.preprocessors.RFE>` refits
after every elimination, which is the more faithful answer and costs one fit per round.

`fit_model` is a `(dataset, features) -> estimator` callable, so this works with a Batcher
estimator, a scikit-learn one, or a closure that fits a whole pipeline:

```python
from batcher.ml import LinearRegression
from batcher.ml.preprocessors import RFE

rfe = RFE(
    lambda d, features: LinearRegression(list(features), "y").fit(d),
    features=["a", "b"],
    n_features=1,
)
print(rfe.fit(regression).selected_)
# ['a']
print(rfe.ranking_["a"])
# 1
```

`ranking_` gives rank 1 to the survivors and a higher rank the earlier a feature was
eliminated, matching scikit-learn's convention. Raise `step` to drop several features per
round when the table is wide enough that one-at-a-time is too many fits; a float is read as
a fraction of the features still in play.

## Composing into a pipeline

Selectors are ordinary preprocessors, so they chain with the rest:

```python
from batcher.ml.preprocessors import Chain, StandardScaler

pipeline = Chain(SelectKBest("y", k=2), StandardScaler(["signal"]))
print(sorted(pipeline.fit_transform(ds).columns))
# ['signal', 'weak', 'y']
```

Put selection early. Everything downstream then runs on a narrower table, which is where
the saving is.

## Requirements and limitations

- A univariate score sees one feature at a time. It cannot detect a feature that matters
  only in combination, and it cannot detect that two features are the same feature — use
  `DropCorrelated` for the latter.
- `chi2` and `mutual_info` reject a feature with one distinct value per row, because a
  contingency statistic sits at its structural maximum there however unrelated the column
  is. Bucket a continuous column first.
- `SelectFromModel` reads coefficient magnitudes, which are only comparable across features
  when the features are on a comparable scale. Scale before you fit the model you read.
- `RFE` costs one model fit per round. On a wide table, filter first and run `RFE` on what
  is left.

## See also

- {doc}`/ml/evaluation/statistics-and-drift` for the scoring functions themselves and the
  wider feature-profiling report.
- {doc}`pipelines` for composing selection with the rest of a feature pipeline.
- {doc}`/api/models/preprocessors` for the full reference.
