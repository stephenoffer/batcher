# Generating features

These preprocessors add columns rather than rewrite existing ones: calendar parts of a
timestamp, text surface statistics, lag and rolling history, curve bases, derived ratios and
group statistics, dimensionality reduction, and finally one assembled feature vector. They
run after encoding and scaling, on columns that are already clean.

```python
import batcher as bt
```

## Features from a timestamp

A raw timestamp is the least useful column in a feature table. A tree model can only split
it into "before and after some instant", which generalizes to nothing. A linear model treats
it as a number that grows forever. The parts repeat, so the parts are what a model learns
from.

{py:class}`DateTimeFeaturizer <batcher.ml.preprocessors.DateTimeFeaturizer>` expands a timestamp into calendar parts as ordinary integer columns.
That suits a tree, which can split on "hour >= 18" directly.

```python
import datetime as dt

import batcher as bt
from batcher.ml.preprocessors import DateTimeFeaturizer

ds = bt.from_pydict({"ordered_at": [dt.datetime(2024, 3, 16, 14, 30)]})
featurized = DateTimeFeaturizer("ordered_at", parts=["hour", "weekday", "is_weekend"])
print(featurized.fit_transform(ds).columns)
```

{py:class}`CyclicalEncoder <batcher.ml.preprocessors.CyclicalEncoder>` is what a linear model, a distance metric, or a neural net needs instead.
As integers, hour 23 and hour 0 sit 23 units apart while being one hour apart. The model
learns a discontinuity at midnight that isn't there, and no amount of scaling fixes it. Two
coordinates on a circle, `_sin` and `_cos`, put them next to each other.

```python
from batcher.ml.preprocessors import CyclicalEncoder

hours = bt.from_pydict({"ordered_at": [dt.datetime(2024, 1, 1, 23), dt.datetime(2024, 1, 2, 0)]})
circle = CyclicalEncoder("ordered_at", parts=["hour"]).fit_transform(hours).to_pydict()
print([round(v, 4) for v in circle["ordered_at_hour_cos"]])
```

Both are stateless. The same expression applies to training and serving data, with nothing
fitted in between.

## Surface features from text

An embedding is the richest way to featurize text, and the most expensive. Plenty of text
signal needs no model at all: whether a review is long, whether a message is all caps, how
many digits a field has. A gradient-boosted model splits on exactly those.
{py:class}`TextStatFeaturizer <batcher.ml.preprocessors.TextStatFeaturizer>` computes them as string expressions, so a dozen text features
over a billion rows is one pass with no GPU.

```python
import batcher as bt
from batcher.ml.preprocessors import TextStatFeaturizer

ds = bt.from_pydict({"review": ["GREAT product!!! 10/10", "ok"]})
out = TextStatFeaturizer("review", features=["char_count", "upper_ratio", "digit_ratio"])
print(out.fit_transform(ds).to_pydict()["review_upper_ratio"])
```

Reach for an embedding when these plateau, not before.

## History as features

A forecasting model needs to know what happened before, and lags and rolling aggregates
carry that. They're also where forecasting pipelines leak most often. A rolling mean that
includes the current row has the target's own value inside its feature, and a "last 7 days"
window computed over the whole table mixes entities together. Both produce a
cross-validated score no deployment reproduces, and neither raises.

So the window of {py:class}`RollingFeaturizer <batcher.ml.preprocessors.RollingFeaturizer>` ends at the previous row, with no option to include the
current one. It and {py:class}`LagFeaturizer <batcher.ml.preprocessors.LagFeaturizer>` both take a `partition_by` that keeps each series
separate.

```python
import batcher as bt
from batcher.ml.preprocessors import LagFeaturizer, RollingFeaturizer

sales = bt.from_pydict({"store": ["a", "a", "a"], "day": [1, 2, 3], "units": [10.0, 20.0, 60.0]})
lagged = LagFeaturizer("units", order_by="day", lags=[1], partition_by="store")
rolled = RollingFeaturizer("units", order_by="day", window=2, partition_by="store")
out = rolled.fit_transform(lagged.fit_transform(sales)).sort("day")
print(out.to_pydict()["units_rolling_mean_2"])
```

Rows near the start of a series have no history and get null. Drop them, or let a booster
use the null as the signal it is.

## Fitting a curve with a linear model

A linear model fits a straight line through each feature, so a relationship that bends has
to arrive as extra columns. Two preprocessors add them, and they behave very differently.

{py:class}`PolynomialFeatures <batcher.ml.preprocessors.PolynomialFeatures>` adds powers and
products. Use it for interactions, where two features act together:

```python
from batcher.ml.preprocessors import PolynomialFeatures

ds = bt.from_pydict({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
print(PolynomialFeatures(["a", "b"], degree=2).fit_transform(ds).columns)
# ['a', 'b', 'a^2', 'a*b', 'b^2']
```

For curvature in a single feature it's usually the wrong tool. A degree-3 polynomial makes
the whole column one cubic, so it oscillates near the edges of the range, and a point at one
end can move the fit at the other.

{py:class}`SplineTransformer <batcher.ml.preprocessors.SplineTransformer>` expands a column
into a B-spline basis instead. Each basis function is non-zero over a few knots only, so the
fit is local: a wiggle in one region stays there. Generalized additive models get their
smooth terms this way.

```python
from batcher.ml.preprocessors import SplineTransformer

curved = bt.from_pydict({"x": [float(i) for i in range(20)]})
spline = SplineTransformer("x", n_knots=5, degree=3).fit(curved)
print([c for c in spline.transform(curved).columns if c.startswith("x_sp")])
# ['x_sp0', 'x_sp1', 'x_sp2', 'x_sp3', 'x_sp4', 'x_sp5', 'x_sp6']
```

The basis has `n_knots + degree - 1` columns. Every row's values sum to one, so the
expansion adds shape without adding scale.

Knots go at the column's quantiles by default, following the data's density rather than its
range, which keeps the basis well-behaved on a skewed column. Pass `knots="uniform"` to space
them evenly across the observed range:

```python
print(SplineTransformer("x", n_knots=3, knots="uniform").fit(curved).knots_["x"])
# [0.0, 9.5, 19.0]
```

A column with a heavy point mass returns the same quantile several times. Repeated knots
would make consecutive basis functions identical, and a linear fit over perfectly collinear
columns is singular. So duplicates collapse, and the basis comes out narrower than
`n_knots` suggests.

## Ratios, interactions, and group statistics

{py:class}`InteractionFeatures <batcher.ml.preprocessors.InteractionFeatures>` appends the pairwise products of its columns, and
{py:class}`RatioFeatures <batcher.ml.preprocessors.RatioFeatures>` appends the ratio of each named pair. A linear model can't build
either from the raw columns:

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

{py:class}`GroupStatEncoder <batcher.ml.preprocessors.GroupStatEncoder>` attaches a per-group statistic of a value column to every row of that
group, so a row can see how it compares to its cohort. `fit` learns the statistics on the
training data and `transform` joins them back on. {py:class}`GroupImputer <batcher.ml.preprocessors.GroupImputer>` fills nulls with the
group's mean rather than the global one, which matters when the groups differ:

```python
from batcher.ml.preprocessors import GroupImputer, GroupStatEncoder

grouped = bt.from_pydict({"grp": ["x", "x", "y", "y"], "val": [1.0, 3.0, 5.0, None]})
encoded = GroupStatEncoder("val", by="grp", statistics=["mean"]).fit_transform(grouped)
print(encoded.collect().column_names)
# ['grp', 'val', 'val_mean_by_grp']
imputed = GroupImputer("val", by="grp").fit_transform(grouped)
print(imputed.collect().column("val").to_pylist())
# [1.0, 3.0, 5.0, 5.0]
```

{py:class}`Binarizer <batcher.ml.preprocessors.Binarizer>` maps a numeric column to 0/1 by a threshold, the "is the balance over the
limit" feature a linear model can't draw for itself. It's stateless:

```python
from batcher.ml.preprocessors import Binarizer

table = bt.from_pydict(
    {"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0], "const": [5.0, 5.0, 5.0, 5.0]}
)
print(Binarizer("a", threshold=2.5).fit_transform(table).collect().column("a").to_pylist())
# [0, 0, 1, 1]
```

## Applying an expression as a pipeline step

{py:class}`FunctionTransformer <batcher.ml.preprocessors.FunctionTransformer>` wraps an
arbitrary expression so it becomes part of the fitted object instead of a loose
`with_columns` call. A chain is applied to the validation split as a unit, and the loose
call is the step that gets forgotten.

`func` receives the column's expression and returns a new one, so the work still runs in
Rust:

```python
from batcher.ml.preprocessors import FunctionTransformer

amounts = bt.from_pydict({"amount": [1.0, 4.0, 9.0]})
print(FunctionTransformer("amount", lambda c: c.sqrt()).fit_transform(amounts).to_pydict())
# {'amount': [1.0, 2.0, 3.0]}
```

Pass `suffix=` to write a new column instead of replacing the old one. A function written
against a value, such as `lambda v: log(v)`, raises a `PlanError` that says so. A chain
holding a lambda can't be saved, so name `func` at module scope if the fitted chain has to
ship.

## Reducing dimensionality

{py:class}`PCA <batcher.ml.preprocessors.PCA>` projects a block of correlated numeric
columns onto their top principal components, replacing them with uncorrelated `pc1`, `pc2`,
... columns ordered by the variance they carry. That removes multicollinearity and shrinks a
wide table, and `explained_variance_ratio_` tells you how many components to keep. The fit
is one scan, because the mean and covariance are mergeable aggregates. Only the small
eigendecomposition runs on the driver.

```python
from batcher.ml.preprocessors import PCA

ds = bt.from_pydict(
    {"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 2.1, 2.9, 4.0], "c": [4.0, 3.0, 2.0, 1.0]}
)
reducer = PCA(["a", "b", "c"], n_components=2).fit(ds)
print(reducer.transform(ds).columns)
# ['pc1', 'pc2']
```

{py:class}`TruncatedSVD <batcher.ml.preprocessors.TruncatedSVD>` skips the centering. Use it
on a non-negative or sparse block, such as a bag-of-words count matrix, where centering
would destroy the structure. On centered data it coincides with `PCA`.

## Random projection

`PCA` finds the directions carrying the most variance, at the price of a covariance pass
and an eigendecomposition over the full width. Often you can't afford that, and often you
don't need it.

{py:class}`GaussianRandomProjection <batcher.ml.preprocessors.GaussianRandomProjection>` and
{py:class}`SparseRandomProjection <batcher.ml.preprocessors.SparseRandomProjection>` multiply
the block through a random matrix instead. The Johnson-Lindenstrauss lemma says that
preserves every pairwise distance to within a small factor, with a bound that depends only
on the target width and the row count, not on the input width or the data:

```python
from batcher.ml.preprocessors import SparseRandomProjection, johnson_lindenstrauss_min_dim

wide = bt.from_pydict({f"f{i}": [float(i + j) for j in range(50)] for i in range(200)})
projected = SparseRandomProjection(list(wide.columns), n_components=64, seed=0)
print(len([c for c in projected.fit_transform(wide).columns if c.startswith("rp")]))
# 64
```

Nothing is read from the data, so these fit on a stream. The matrix depends only on the
seed and the input width, so training and serving can't disagree. Size the target with
`johnson_lindenstrauss_min_dim` rather than guessing:

```python
print(johnson_lindenstrauss_min_dim(10_000, eps=0.2))
# 2125
```

Prefer the sparse variant on a wide block. Most of its matrix is zero, and zero entries are
left out of the lowered expression, so the engine evaluates a much smaller expression per
row for the same distance guarantee.

## Kernel features for a linear model

A kernel SVM is often the best model on a medium tabular problem, and the worst thing to put
in a pipeline. It needs the full pairwise kernel matrix, which is quadratic in rows and
doesn't distribute.

{py:class}`RBFSampler <batcher.ml.preprocessors.RBFSampler>` and
{py:class}`Nystroem <batcher.ml.preprocessors.Nystroem>` map each row into a space where an
ordinary dot product approximates the RBF kernel. Fit a plain linear model on those columns
and you get most of the kernel's accuracy from something that streams and distributes:

```python
from batcher.ml.preprocessors import Chain, RBFSampler, StandardScaler

points = bt.from_pydict({"a": [0.0, 1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0, 0.0]})
mapped = Chain(
    StandardScaler(["a", "b"]),
    RBFSampler(["a", "b"], n_components=64, gamma=0.5, seed=0),
).fit_transform(points)
print(len([c for c in mapped.columns if c.startswith("rbf")]))
# 64
```

Scale first, as above. `gamma` is a distance in feature space, and a column measured in
millions can't share a sensible value with one measured in fractions.

`RBFSampler` draws its map from a seed and reads no data, so it works on a stream.
`Nystroem` picks `n_components` actual rows as landmarks and measures similarity to them.
That's data-dependent, so it usually needs fewer components for the same accuracy, at the
cost of a fit pass.

## Assembling features

{py:class}`Concatenator <batcher.ml.preprocessors.Concatenator>` stacks numeric columns into one list column, the feature vector a trainer
reads. It's stateless, but it still follows the fit/transform contract, so call
`fit_transform`, or `fit` then `transform`. The source columns are kept unless `drop=True`.

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
{doc}`/ml/training/data-loaders`.

## Pinning the feature contract

A trained model is valid only against the exact columns, order, and dtypes it saw in
training. {py:class}`FeatureSpec <batcher.ml.FeatureSpec>` records that contract. Build it from the training frame with
`from_dataset`, then `align` a scoring frame to the pinned order, or `validate` it to raise
on a missing, reordered, or retyped column rather than scoring wrong numbers:

```python
from batcher.ml import FeatureSpec

train = bt.from_pydict({"age": [30.0, 40.0], "income": [50.0, 60.0], "label": [0, 1]})
spec = FeatureSpec.from_dataset(train, features=["age", "income"])
print(spec.dtypes)
# {'age': 'double', 'income': 'double'}
scoring = bt.from_pydict({"income": [55.0], "age": [35.0], "id": [7]})
print(spec.align(scoring).columns)
# ['age', 'income']
```

Dtype names are the engine's own, as `Dataset.dtypes` renders them, so a float column pins
as `double`, not `float64`.

## See also

- {doc}`/ml/preparing/preprocessors/pipelines`: sequencing these steps and saving the fitted result.
- {doc}`/ml/preparing/preprocessors/encoding`: making a categorical column numeric before it feeds these.
- {doc}`/ml/preparing/preprocessors/feature-selection`: dropping the columns that don't earn their place, including `VarianceThreshold`.
- {doc}`/ml/preparing/tokenization`: {py:class}`Tokenizer <batcher.ml.preprocessors.Tokenizer>`, sequence packing, and label encoding for text.
- {doc}`/user-guide/transform/columns/expression-recipes`: the same feature work written by hand as expressions.
