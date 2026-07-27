# Generating features

This page covers the preprocessors that *add* columns rather than rewrite existing
ones: timestamp expansion, text surface statistics, lag and rolling history,
dimensionality reduction, and assembling the result into a single tensor column.

These run after the encoding and scaling steps, on columns that are already clean.

```python
import batcher as bt
```

## Features from a timestamp

A raw timestamp is the least useful column in a feature table. A tree model can only split
it into "before and after some instant", which generalizes to nothing; a linear model treats
it as a number that grows forever. What a model can learn from is the *parts*, because those
repeat.

`DateTimeFeaturizer` expands a timestamp into calendar parts as ordinary integer columns,
which is what a tree wants — it can split on "hour >= 18" directly.

```python
import datetime as dt

import batcher as bt
from batcher.ml.preprocessors import DateTimeFeaturizer

ds = bt.from_pydict({"ordered_at": [dt.datetime(2024, 3, 16, 14, 30)]})
featurized = DateTimeFeaturizer("ordered_at", parts=["hour", "weekday", "is_weekend"])
print(featurized.fit_transform(ds).columns)
```

`CyclicalEncoder` is what a linear model, a distance metric, or a neural net needs instead.
Encoded as an integer, hour 23 and hour 0 sit 23 units apart while being one hour apart, so
the model learns a discontinuity at midnight that is not there — and no amount of scaling
fixes it. Two coordinates on a circle put them adjacent, which is the truth.

```python
from batcher.ml.preprocessors import CyclicalEncoder

hours = bt.from_pydict(
    {"ordered_at": [dt.datetime(2024, 1, 1, 23), dt.datetime(2024, 1, 2, 0)]}
)
circle = CyclicalEncoder("ordered_at", parts=["hour"]).fit_transform(hours).to_pydict()
print([round(v, 4) for v in circle["ordered_at_hour_cos"]])
```

Both are stateless, so the same expression applies to training and serving data with nothing
fitted in between.

## Surface features from text

An embedding is the powerful way to featurize text and the expensive one. A great many text
signals need no model at all — whether a review is long, whether a message is all-caps, how
many digits a field has — and they are what a gradient-boosted model actually splits on.
`TextStatFeaturizer` computes them as pure string expressions, so a dozen text features over
a billion rows is one pass and no GPU.

```python
import batcher as bt
from batcher.ml.preprocessors import TextStatFeaturizer

ds = bt.from_pydict({"review": ["GREAT product!!! 10/10", "ok"]})
out = TextStatFeaturizer("review", features=["char_count", "upper_ratio", "digit_ratio"])
print(out.fit_transform(ds).to_pydict()["review_upper_ratio"])
```

Reach for an embedding when these plateau, not before.

## History as features

A forecasting model needs to know what happened before, and the columns that carry that are
lags and rolling aggregates. Building them is where forecasting pipelines leak most often:
a rolling mean that includes the current row has the target's own value inside its own
feature, and a "last 7 days" window computed over the whole table mixes entities together.
Both produce a cross-validated score no deployment reproduces, and neither raises.

`RollingFeaturizer`'s window therefore ends at the **previous** row by construction, with no
option to include the current one, and both take a `partition_by` that keeps each series
separate.

```python
import batcher as bt
from batcher.ml.preprocessors import LagFeaturizer, RollingFeaturizer

sales = bt.from_pydict(
    {"store": ["a", "a", "a"], "day": [1, 2, 3], "units": [10.0, 20.0, 60.0]}
)
lagged = LagFeaturizer("units", order_by="day", lags=[1], partition_by="store")
rolled = RollingFeaturizer("units", order_by="day", window=2, partition_by="store")
out = rolled.fit_transform(lagged.fit_transform(sales)).sort("day")
print(out.to_pydict()["units_rolling_mean_2"])
```

Rows near the start of a series have no history and get null, which is the honest answer:
drop them, or let a booster use the null as the signal it is.

## Reducing dimensionality

`PCA` projects a block of correlated numeric columns onto their top principal components, replacing them with a few uncorrelated `pc1`, `pc2`, ... columns ordered by the variance they carry. It kills multicollinearity, shrinks a wide table for a downstream model, and its `explained_variance_ratio_` tells you how many components to keep. The fit is a single scan — the mean and covariance are aggregates — and only the small eigendecomposition runs on the driver.

```python
from batcher.ml.preprocessors import PCA

ds = bt.from_pydict({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 2.1, 2.9, 4.0], "c": [4.0, 3.0, 2.0, 1.0]})
reducer = PCA(["a", "b", "c"], n_components=2).fit(ds)
print(reducer.transform(ds).columns)
# ['pc1', 'pc2']
```

`TruncatedSVD` is the same idea without centering the columns first, which is what you want on a non-negative or sparse feature block (a bag-of-words count matrix) where centering would destroy the structure. On centered data it coincides with `PCA`.

## Assembling features

`Concatenator` stacks several numeric columns into one list column. It is the "make a
feature vector" step before training. It is stateless, so `fit` is a no-op, but it
follows the contract, so use `fit_transform` or `fit` then `transform`. The source
columns are kept unless `drop=True`.

```python
import batcher as bt
from batcher.ml.preprocessors import Concatenator

ds = bt.from_pydict({"age": [20.0, 30.0], "score": [1.0, 2.0]})
assembled = Concatenator(["age", "score"], output_column="features").fit_transform(ds).collect()
print(assembled.column_names)
# ['age', 'score', 'features']
print(assembled.column("features").to_pylist())
# [[20.0, 1.0], [30.0, 2.0]]
```

The assembled list column becomes a tensor for training with zero or one copy. See
[PyTorch integration](../pytorch.md).

### Deriving and selecting columns

Before assembly, you often build new columns and drop useless ones. `InteractionFeatures`
appends the pairwise products of its columns, and `RatioFeatures` appends the ratio of
each named pair. Both give a linear model a signal it cannot learn from the raw columns:

```python
import batcher as bt
from batcher.ml.preprocessors import InteractionFeatures, RatioFeatures

ds = bt.from_pydict({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
crossed = InteractionFeatures(["a", "b"]).fit_transform(ds)
print(crossed.collect().column_names)
# ['a', 'b', 'a_x_b']
ratioed = RatioFeatures([("a", "b")]).fit_transform(ds)
print(ratioed.collect().column_names)
# ['a', 'b', 'a_per_b']
```

`GroupStatEncoder` attaches a per-group statistic of a value column to every row of that
group, so a row can see how it compares to its cohort. `GroupImputer` fills nulls with the
group's mean rather than the global one, which matters when the groups differ:

```python
from batcher.ml.preprocessors import GroupImputer, GroupStatEncoder

grouped = bt.from_pydict(
    {"grp": ["x", "x", "y", "y"], "val": [1.0, 3.0, 5.0, None]}
)
encoded = GroupStatEncoder("val", by="grp", statistics=["mean"]).fit_transform(grouped)
print(encoded.collect().column_names)
# ['grp', 'val', 'val_mean_by_grp']
imputed = GroupImputer("val", by="grp").fit_transform(grouped)
print(imputed.collect().column("val").to_pylist())
# [1.0, 3.0, 5.0, 5.0]
```

`Binarizer` maps a numeric column to 0/1 by a threshold. `VarianceThreshold` drops columns
whose variance is at or below a threshold, which removes constant columns for free.
`ColumnSelector` and `ColumnDropper` are the projection steps as pipeline stages, for when
selection has to sit inside a `Chain`:

```python
from batcher.ml.preprocessors import (
    Binarizer,
    ColumnDropper,
    ColumnSelector,
    VarianceThreshold,
)

table = bt.from_pydict(
    {"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0], "const": [5.0, 5.0, 5.0, 5.0]}
)
print(Binarizer("a", threshold=2.5).fit_transform(table).collect().column("a").to_pylist())
# [0, 0, 1, 1]
print(VarianceThreshold("const").fit_transform(table).collect().column_names)
# ['a', 'b']
print(ColumnSelector(["a", "b"]).fit_transform(table).collect().column_names)
# ['a', 'b']
print(ColumnDropper(["const"]).fit_transform(table).collect().column_names)
# ['a', 'b']
```

### Pinning the feature contract

A trained model is valid only against the exact columns, order, and dtypes it saw during
training. `FeatureSpec` captures that contract so a scoring frame can be checked against it
rather than producing wrong numbers on a silently reordered or retyped input:

```python
from batcher.ml import FeatureSpec

spec = FeatureSpec(["age", "income"], {"age": "float64", "income": "float64"})
print(spec.features)
# ['age', 'income']
print(spec.dtypes)
# {'age': 'float64', 'income': 'float64'}
```

`Tokenizer` maps a text column through a user-supplied tokenizer, which is either a
`str -> list` callable or any object with `.encode`, such as a HuggingFace tokenizer.
Tokenization is inherently per-string, so it runs as a whole-batch `map_batches` UDF. It
needs a real tokenizer, so it is shown but not run here.

```python
# docs: skip
from batcher.ml.preprocessors import Tokenizer
from transformers import AutoTokenizer

hf = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenized = Tokenizer("text", hf, output_column="input_ids").fit_transform(ds)
```

## See also

:::{seealso}
- {doc}`pipelines`: sequencing these steps and saving the fitted result.
- {doc}`encoding`: making a categorical column numeric before it feeds these.
- {doc}`../../user-guide/expression-recipes`: the same feature work written by hand as expressions.
:::
