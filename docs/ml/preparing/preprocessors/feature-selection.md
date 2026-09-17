# Feature selection

This page covers the selectors that cut a wide feature table down, and how to use them
without leaking the validation split into the choice.

A column with no signal still costs memory, still costs a shuffle when you join, and still
gives a model one more chance to fit noise. Two columns that duplicate each other are worse.
A linear fit has to split one effect between them, so the coefficients come out large,
opposite in sign, and unstable under resampling.

## Selection is fitted state

A selector is an object that holds its decision. Choose features on the whole frame, or
re-choose them per split, and the held-out rows have taken part in the choice. Your
validation score is then optimistic by an amount nothing measures.

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

Work down this table, stopping when the feature set is narrow enough. Each row costs
more than the one above it:

| Selector | Cost | Sees interaction |
|---|---|---|
| {py:class}`VarianceThreshold <batcher.ml.preprocessors.VarianceThreshold>` | One variance aggregate, no target | No |
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

A univariate score is blind to interaction. A feature that only matters alongside another
scores as noise, so use this to remove obvious dead weight, not to have the final word.

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
from every pair correlated above `threshold`. Position in the column list decides which one
goes, so two runs over the same table produce the same feature set:

```python
from batcher.ml.preprocessors import DropCorrelated

duplicated = bt.from_pydict(
    {"a": [1.0, 2.0, 3.0, 4.0], "a_copy": [2.0, 4.0, 6.0, 8.0], "b": [1.0, 0.0, 1.0, 0.0]}
)
print(DropCorrelated(threshold=0.95).fit(duplicated).dropped_)
# ['a_copy']
```

Pass `keep` for a column that must survive whatever it correlates with. Its partner goes
instead, and the pair isn't left standing:

```python
print(DropCorrelated(keep=["a_copy"]).fit(duplicated).dropped_)
# ['a']
```

## Dropping columns without a target

{py:class}`VarianceThreshold <batcher.ml.preprocessors.VarianceThreshold>` needs no target. It drops each column whose variance is at or
below `threshold`, which removes constant columns for the price of one aggregate.
{py:class}`ColumnSelector <batcher.ml.preprocessors.ColumnSelector>` and {py:class}`ColumnDropper <batcher.ml.preprocessors.ColumnDropper>` are plain projections as pipeline
stages, for when a hand-picked selection has to sit inside a {py:class}`Chain <batcher.ml.preprocessors.Chain>`:

```python
from batcher.ml.preprocessors import ColumnDropper, ColumnSelector, VarianceThreshold

table = bt.from_pydict(
    {"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0], "const": [5.0, 5.0, 5.0, 5.0]}
)
print(VarianceThreshold("const").fit_transform(table).collect().column_names)
# ['a', 'b']
print(ColumnSelector(["a", "b"]).fit_transform(table).collect().column_names)
# ['a', 'b']
print(ColumnDropper(["const"]).fit_transform(table).collect().column_names)
# ['a', 'b']
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

The estimator must already be fitted. Refitting inside the selector would hide which rows
the selection saw, and that decides whether it leaks.

`threshold` also accepts `"mean"` or `"median"` to cut at that statistic of the
importances, and `max_features` caps the count regardless.

{py:func}`feature_importances <batcher.ml.feature_importances>` is what the selector reads.
Call it directly to see the magnitudes rather than only the survivors. It understands
Batcher's estimators, scikit-learn's `feature_importances_` and `coef_`, and a plain dict:

```python
from batcher.ml.preprocessors import feature_importances

print(sorted(feature_importances(model)))
# ['a', 'b']
```

Coefficient magnitudes are only comparable when the features are on a comparable scale.
Scale first, then fit the model you read.

## Recursive elimination

Dropping a feature changes what the survivors are worth, so ranking once and cutting isn't
the same as cutting one at a time. {py:class}`RFE <batcher.ml.preprocessors.RFE>` refits
after every elimination. That's the more faithful answer, at one fit per round.

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
round when one at a time is too many fits. A float is read as a fraction of the features
still in play.

## Composing into a pipeline

Selectors are ordinary preprocessors, so they chain with the rest. Fit the chain on the
training split, as you would fit the selector alone:

```python
from batcher.ml.preprocessors import Chain, StandardScaler

pipeline = Chain(SelectKBest("y", k=2), StandardScaler(["signal"])).fit(train)
print(sorted(pipeline.transform(test).columns))
# ['signal', 'weak', 'y']
```

Put selection early, so everything downstream runs on the narrower table.

## Requirements and limitations

- A univariate score sees one feature at a time. It cannot detect a feature that matters
  only in combination, and it cannot detect that two features are the same feature. Use
  `DropCorrelated` for the latter.
- `chi2` and `mutual_info` reject a feature with one distinct value per row, because a
  contingency statistic sits at its structural maximum there however unrelated the column
  is. Bucket a continuous column first.
- `RFE` costs one model fit per round. On a wide table, filter first and run `RFE` on what
  is left.

## See also

- {doc}`/ml/evaluation/statistics-and-drift`: the scoring functions themselves and the wider feature-profiling report.
- {doc}`/ml/preparing/preprocessors/feature-generation`: the steps that widen the table this page narrows.
- {doc}`/ml/preparing/preprocessors/pipelines`: composing selection with the rest of a feature pipeline.
- {doc}`/ml/evaluation/model-selection`: cross-validating the pipeline a selector sits in.
- {doc}`/api/models/preprocessors`: the full reference.
