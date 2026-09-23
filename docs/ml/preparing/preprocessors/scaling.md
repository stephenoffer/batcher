# Scaling and distributions

Two different problems hide under "scale this column". If features sit on different
scales and the largest one dominates a distance or a gradient step, you want a scaler. If a
column is heavily skewed, a scaler won't help. It moves the numbers and leaves the shape
where it was, so you want a distribution reshaper instead. This page covers both, plus the
row-wise normalizer and the rank and label transforms.

## Scaling numeric columns

A scaler learns summary statistics in `fit` and rewrites each column in place. The columns
named in the constructor are replaced. The rest of the dataset passes through untouched.

```python
import batcher as bt
from batcher.ml.preprocessors import StandardScaler

train = bt.from_pydict({"age": [20.0, 30.0, 40.0, 50.0], "score": [1.0, 2.0, 3.0, 4.0]})

scaler = StandardScaler(["age", "score"]).fit(train)
scaled = scaler.transform(train).collect()
print([round(v, 3) for v in scaled.column("age").to_pylist()])
# [-1.342, -0.447, 0.447, 1.342]
```

The fitted statistics live on the object, so the same scaler standardizes a held-out split
with the training mean and standard deviation. Never refit on validation data. The two
splits would then sit on different scales, and nothing would tell you:

```python
val = bt.from_pydict({"age": [35.0], "score": [2.5]})
print(scaler.transform(val).collect().column("age").to_pylist())
# [0.0]  (35.0 is the training mean, so it standardizes to zero)
```

{py:class}`MinMaxScaler <batcher.ml.preprocessors.MinMaxScaler>` maps each column into `feature_range`, which defaults to `[0, 1]`, by its
learned min and max. Pass `feature_range=(lo, hi)` for another target interval.
{py:class}`MaxAbsScaler <batcher.ml.preprocessors.MaxAbsScaler>` divides by the maximum absolute value into `[-1, 1]` without centering, so
it preserves sparsity. {py:class}`RobustScaler <batcher.ml.preprocessors.RobustScaler>` centers on the median and divides by the
interquartile range, so a few outliers do not dominate the scale.

```python
import batcher as bt
from batcher.ml.preprocessors import MinMaxScaler, MaxAbsScaler, RobustScaler

ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})

print(
    [round(v, 3) for v in MinMaxScaler(["x"]).fit_transform(ds).collect().column("x").to_pylist()]
)
# [0.0, 0.25, 0.5, 0.75, 1.0]
print(MaxAbsScaler(["x"]).fit_transform(ds).collect().column("x").to_pylist())
# [0.2, 0.4, 0.6, 0.8, 1.0]
print(RobustScaler(["x"]).fit_transform(ds).collect().column("x").to_pylist())
# [-1.0, -0.5, 0.0, 0.5, 1.0]
```

No scaler divides a constant column by zero. `StandardScaler` and `RobustScaler` fall back
to a scale of 1.0, so the column comes out centered. `MaxAbsScaler` leaves an all-zero
column as it is, and `MinMaxScaler` maps a constant column to the bottom of
`feature_range`. You get finite values, never NaN.

### Normalizing per row

{py:class}`Normalizer <batcher.ml.preprocessors.Normalizer>` is the row-wise scaler. It divides each row by its norm across the
named columns, so every row comes out a unit vector. Nothing is learned, so `transform`
works straight after construction with no `fit`. The default `norm="l2"` divides by the
square root of the sum of squares, `"l1"` by the sum of absolute values, and `"max"` by the
largest absolute value.

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

Scaling changes a column's units. The transforms in this section change its *shape*, and
shape is often what a linear model, a distance metric, or a neural net needs fixed.
Standardize a log-normal column and it comes out just as skewed as it went in.

{py:class}`QuantileTransformer <batcher.ml.preprocessors.QuantileTransformer>` is the most aggressive and the most reliable. It keeps only the
*order* of the values, learning `n_quantiles` cut points in one aggregate, so the output is
uniform whatever went in and an outlier can't survive it. Pass
`output_distribution="normal"` for a standard-normal shape instead. The mapping is a step
function with `n_quantiles` steps, not an interpolation.

```python
import batcher as bt
from batcher.ml.preprocessors import QuantileTransformer

ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 1000.0]})
print(QuantileTransformer("x", n_quantiles=4).fit_transform(ds).to_pydict())
```

{py:class}`PowerTransformer <batcher.ml.preprocessors.PowerTransformer>` is the data-driven middle ground. It picks the Yeo-Johnson power
that makes the column most Gaussian by maximum likelihood. The likelihood at every
candidate lambda is an aggregate, so one pass evaluates a whole grid of 41 candidates rather
than one scan per optimizer iteration. A coarse grid over `[-2, 2]` and two zoomed grids
around its best point take three passes and land within about 1e-4 of scikit-learn's
lambda. The standardization that follows uses the population standard deviation, as
scikit-learn's does.

```python
from batcher.ml.preprocessors import PowerTransformer

skewed = bt.from_pydict({"x": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]})
print(PowerTransformer("x").fit(skewed).lambdas_["x"] < 0.5)
```

{py:class}`BoxCoxTransformer <batcher.ml.preprocessors.BoxCoxTransformer>` fits the same way on the Box-Cox family. It needs strictly positive values and raises on anything else rather than producing NaNs. Use it to reproduce an existing Box-Cox analysis, and `PowerTransformer` when the column can be zero or negative.

```python
from batcher.ml.preprocessors import BoxCoxTransformer

positive = bt.from_pydict({"x": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]})
print(-2.0 <= BoxCoxTransformer("x").fit(positive).lambdas_["x"] <= 2.0)
```

{py:class}`LogTransformer <batcher.ml.preprocessors.LogTransformer>` is the one you can explain to a stakeholder. `log1p` suits a
multiplicative quantity, it's stateless, and nobody has to ask what a lambda of 0.3 means.

{py:class}`Clipper <batcher.ml.preprocessors.Clipper>` clamps into a learned quantile range instead of dropping rows, so the row count
and every join key survive. The training cut points apply to serving data too. A new
record-breaking value gets clamped, not extrapolated into a region the model never saw.

{py:class}`MissingIndicator <batcher.ml.preprocessors.MissingIndicator>` records which values were missing *before* an imputer fills them.
Missingness is often a signal. A blank income field means something different from a low
one, and once the imputer has run the distinction is gone.

```python
from batcher.ml.preprocessors import Chain, MissingIndicator, SimpleImputer

ds = bt.from_pydict({"income": [50000.0, None, 70000.0]})
flagged = Chain(MissingIndicator("income"), SimpleImputer(["income"])).fit_transform(ds)
print(flagged.to_pydict()["income_missing"])
```

## Rank and label transforms

{py:class}`RankTransformer <batcher.ml.preprocessors.RankTransformer>` replaces a value with its percentile rank in `[0, 1]`. Like
`QuantileTransformer` it keeps only the order and ignores outliers. Unlike it, the rank is
exact rather than binned, which matters on a small column.

The rank has no fitted state. It is computed within the frame being transformed, so a
serving row is ranked against its serving batch, not against the training set. Use it on a
whole dataset, not as a train/serve transform.

```python
import batcher as bt
from batcher.ml.preprocessors import RankTransformer

ds = bt.from_pydict({"x": [10.0, 40.0, 20.0, 1000.0]})
print(RankTransformer("x").fit_transform(ds).to_pydict()["x"])
```

{py:class}`LabelBinarizer <batcher.ml.preprocessors.LabelBinarizer>` expands a categorical *label* into one 0/1 column per class, one-vs-rest. It is
the target-side counterpart of one-hot encoding, for a per-class metric or a set of binary
models. {py:class}`MultiLabelBinarizer <batcher.ml.preprocessors.MultiLabelBinarizer>` does the same for a list column where a row carries
several labels at once, such as tags or genres. That is the usual input shape for a
multi-label classifier.

## See also

- {doc}`/ml/preparing/preprocessors/encoding`: the categorical half of the same job, plus imputation.
- {doc}`/ml/preparing/preprocessors/pipelines`: chaining a scaler behind an imputer and saving the fitted state.
- {doc}`/ml/preparing/preprocessors/index`: the fit/transform contract and the full preprocessor table.
- {doc}`/ml/evaluation/statistics-and-drift`: the statistics that tell you which transform a column needs.
