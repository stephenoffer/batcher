# Scaling and distributions

This page covers the preprocessors that change a numeric column's *scale* or its
*shape*: the four scalers, the row-wise normalizer, the distribution reshapers, and the
rank and label transforms.

Reach for a scaler when features are on different scales and a distance or gradient
step would otherwise be dominated by the largest one. Reach for a distribution reshaper
when a column is heavily skewed and a linear rescale would leave it that way.

## Scaling numeric columns

A scaler learns summary statistics in `fit` and rewrites each column in place. The
columns named in the constructor are replaced, and the rest of the dataset passes
through.

```python
import batcher as bt
from batcher.ml.preprocessors import StandardScaler

train = bt.from_pydict({"age": [20.0, 30.0, 40.0, 50.0], "score": [1.0, 2.0, 3.0, 4.0]})

scaler = StandardScaler(["age", "score"]).fit(train)
scaled = scaler.transform(train).collect()
print([round(v, 3) for v in scaled.column("age").to_pylist()])
# [-1.342, -0.447, 0.447, 1.342]
```

The fitted statistics live on the object, so the *same* scaler standardizes a
held-out split with the training mean and standard deviation. Never refit on
validation data, or the splits no longer share a scale:

```python
val = bt.from_pydict({"age": [35.0], "score": [2.5]})
print(scaler.transform(val).collect().column("age").to_pylist())
# [0.0] — 35.0 is the training mean, so it standardizes to zero
```

`MinMaxScaler` maps each column into `feature_range`, which defaults to `[0, 1]`, by its
learned min and max. Pass `feature_range=(lo, hi)` for another target interval.
`MaxAbsScaler` divides by the maximum absolute value into `[-1, 1]` without centering, so
it preserves sparsity. `RobustScaler` centers on the median and divides by the
interquartile range, so a few outliers do not dominate the scale.

```python
import batcher as bt
from batcher.ml.preprocessors import MinMaxScaler, MaxAbsScaler, RobustScaler

ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})

print([round(v, 3) for v in MinMaxScaler(["x"]).fit_transform(ds).collect().column("x").to_pylist()])
# [0.0, 0.25, 0.5, 0.75, 1.0]
print(MaxAbsScaler(["x"]).fit_transform(ds).collect().column("x").to_pylist())
# [0.2, 0.4, 0.6, 0.8, 1.0]
print(RobustScaler(["x"]).fit_transform(ds).collect().column("x").to_pylist())
# [-1.0, -0.5, 0.0, 0.5, 1.0]
```

A constant column with zero variance, zero range, or zero IQR is never divided by zero.
The scaler falls back to a scale of 1.0, or maps to the bottom of `feature_range` for
`MinMaxScaler`, so the column survives the transform unchanged.

### Normalizing per row

`Normalizer` is the row-wise scaler. It divides each row by its norm across the named
columns, so every row becomes a unit vector. It is stateless, with nothing to
learn, but it still follows the `fit` and `transform` contract, so use `transform`
directly after construction, or `fit_transform`. The default `norm="l2"` divides by the
square root of the sum of squares, `"l1"` by the sum of absolute values, and `"max"` by
the largest absolute value.

```python
import batcher as bt
from batcher.ml.preprocessors import Normalizer

ds = bt.from_pydict({"a": [3.0, 1.0], "b": [4.0, 0.0]})
normalized = Normalizer(["a", "b"], norm="l2").transform(ds).collect()
print(normalized.column("a").to_pylist())
# [0.6, 1.0]
print(normalized.column("b").to_pylist())
# [0.8, 0.0]
```

## Reshaping a distribution

Scaling changes a column's units; these change its *shape*, which is what a linear model,
a distance metric, and a neural net actually need. Standardizing a log-normal column leaves
it just as skewed, with a mean still sitting at the 70th percentile.

`QuantileTransformer` is the most aggressive and the most reliable: it keeps only the
*order* of the values, so the output is uniform whatever went in and an outlier cannot
survive it.

```python
import batcher as bt
from batcher.ml.preprocessors import QuantileTransformer

ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 1000.0]})
print(QuantileTransformer("x", n_quantiles=4).fit_transform(ds).to_pydict())
```

`PowerTransformer` is the data-driven middle ground. It finds the Yeo-Johnson power that
makes the column most Gaussian by maximum likelihood, and it does so in **one pass**, because
the likelihood at every candidate lambda is an aggregate, so the whole grid is evaluated
together rather than one scan per optimizer iteration.

```python
from batcher.ml.preprocessors import PowerTransformer

skewed = bt.from_pydict({"x": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]})
print(PowerTransformer("x").fit(skewed).lambdas_["x"] < 0.5)
```

`BoxCoxTransformer` fits the same way on the Box-Cox family, which is what most statistics tooling means by "the Box-Cox transform". It needs strictly positive values and raises on anything else rather than producing NaNs, so use it when reproducing an existing Box-Cox analysis and `PowerTransformer` when the column can be zero or negative.

```python
from batcher.ml.preprocessors import BoxCoxTransformer

positive = bt.from_pydict({"x": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]})
print(-2.0 <= BoxCoxTransformer("x").fit(positive).lambdas_["x"] <= 2.0)
```

`LogTransformer` is the version an analyst can defend to a stakeholder: `log1p` is exactly
right for a multiplicative quantity, it is stateless, and it needs no explanation of what a
lambda of 0.3 means.

`Clipper` clamps into a learned quantile range rather than dropping anything, so the row
count and every join key survive. Applying the *training* cut points to serving data is the
point: a new record-breaking value is clamped rather than extrapolated into a region the
model never saw.

`MissingIndicator` records which values were missing **before** an imputer fills them.
Missingness is usually a signal, since a blank income field means something different from a
low one, and imputing first destroys that signal permanently.

```python
from batcher.ml.preprocessors import Chain, MissingIndicator, SimpleImputer

ds = bt.from_pydict({"income": [50000.0, None, 70000.0]})
flagged = Chain(MissingIndicator("income"), SimpleImputer(["income"])).fit_transform(ds)
print(flagged.to_pydict()["income_missing"])
```

## Rank and label transforms

`RankTransformer` replaces a value with its percentile rank. Like `QuantileTransformer` it
keeps only the order and is immune to outliers, but it is exact (every distinct value gets its
own rank) rather than binned, which matters on a small column.

```python
import batcher as bt
from batcher.ml.preprocessors import RankTransformer

ds = bt.from_pydict({"x": [10.0, 40.0, 20.0, 1000.0]})
print(RankTransformer("x").fit_transform(ds).to_pydict()["x"])
```

`LabelBinarizer` one-vs-rest expands a categorical *label* into a 0/1 column per class. It is the
target-side counterpart of one-hot encoding, for a per-class metric or a set of binary models.
`MultiLabelBinarizer` does the same for a *list* column, where a row can carry many labels at
once (tags, genres), which is the standard input shaping for a multi-label classifier.

## See also

- {doc}`/ml/preparing/preprocessors/encoding`: the categorical half of the same job.
- {doc}`/ml/preparing/preprocessors/index`: the fit/transform contract and the full preprocessor table.
- {doc}`/ml/evaluation/statistics-and-drift`: the statistics that tell you which transform a column needs.
