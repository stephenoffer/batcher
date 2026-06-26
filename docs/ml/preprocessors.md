# Preprocessors

Preprocessors are scikit-learn-style `fit`/`transform` feature transformers that run
on the engine. `fit` learns its state with one mergeable aggregate over the data (so
it is distributed and spillable for free); `transform` is a lazy column rewrite. Fit
on the training set, then `transform` the training **and** validation sets with the
same learned state.

Compose several preprocessors by sequencing them: fit each on the previous step's
output, then transform any split through the same fitted objects.

```python
from batcher.ml.preprocessors import StandardScaler, SimpleImputer
import batcher as bt

train = bt.from_pydict({"age": [20.0, 30.0, 40.0, 50.0], "income": [1.0, 2.0, 3.0, 4.0]})

imputer = SimpleImputer(["age"])
scaler = StandardScaler(["age", "income"])
train_scaled = scaler.fit_transform(imputer.fit_transform(train))
print(train_scaled.collect().column_names)
# ['age', 'income']
```

Because the fitted state lives on each object, the same steps transform held-out
data with the statistics learned on train:

```python
import batcher as bt
from batcher.ml.preprocessors import StandardScaler

train = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
scaler = StandardScaler(["x"]).fit(train)

test = bt.from_pydict({"x": [6.0, 7.0]})
print([round(v, 3) for v in scaler.transform(test).collect().column("x").to_pylist()])
# [2.121, 2.828]
```

## The three-call contract

Every preprocessor exposes the same `Preprocessor` API:

- **`fit(ds)`** runs a small aggregate, stores the learned state on the object, and
  returns `self` (so a stateless transform — `Normalizer`, `Concatenator`,
  `Tokenizer` — also needs a `fit` or `fit_transform` before `transform`).
- **`transform(ds)`** returns a new lazy `Dataset` with the learned rewrite applied;
  it runs no work until a terminal op (`collect`, `write.parquet`, ...).
- **`fit_transform(ds)`** is `fit(ds).transform(ds)` — the common single-split path.

`fit` *executes* (it is the one place a preprocessor touches data); `transform` stays
lazy, so it composes with the rest of the pipeline and runs inside the engine. Calling
`transform` before `fit` raises `PlanError`.

## Available preprocessors

| Class | `fit` learns | `transform` |
| --- | --- | --- |
| `StandardScaler` | mean, population std | `(x - mean) / std` |
| `MinMaxScaler` | min, max | scale into `feature_range` (default `[0, 1]`) |
| `MaxAbsScaler` | max absolute value | `x / max(\|x\|)` into `[-1, 1]` |
| `RobustScaler` | median, IQR | `(x - median) / IQR` (outlier-robust) |
| `OrdinalEncoder` | sorted categories | integer code per category |
| `LabelEncoder` | sorted classes | integer code for one target column |
| `OneHotEncoder` | categories | one 0/1 indicator column per category |
| `MultiHotEncoder` | distinct list elements | one 0/1 indicator column per category, for a list column |
| `KBinsDiscretizer` | bin edges (quantile or uniform) | integer bin index `0..n_bins-1` |
| `Normalizer` | — (stateless) | scale each row to unit L1/L2/max norm across columns |
| `SimpleImputer` | mean / median / mode / constant | fill nulls |
| `Concatenator` | — (stateless) | stack columns into one tensor column |
| `Tokenizer` | — (stateless) | tokenize text with a user tokenizer |

All preprocessors share the `Preprocessor` base contract (`fit` / `transform` /
`fit_transform`).

Each scaler matches scikit-learn's definitions (`StandardScaler` uses population
variance). `fit` lowers to the existing `group_by().agg(...)` and `distinct()`
operators, so it is partition-independent: a fit on a distributed dataset learns the
same statistics as a single-node fit.

## Scaling numeric columns

A scaler learns summary statistics in `fit` and rewrites each column in place. The
columns named in the constructor are replaced; the rest of the dataset passes
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
held-out split with the training mean and standard deviation — never refit on
validation data, or the splits no longer share a scale:

```python
val = bt.from_pydict({"age": [35.0], "score": [2.5]})
print(scaler.transform(val).collect().column("age").to_pylist())
# [-0.4472135954999579]
```

`MinMaxScaler` maps each column into `feature_range` (default `[0, 1]`) by its learned
min and max; pass `feature_range=(lo, hi)` for another target interval. `MaxAbsScaler`
divides by the maximum absolute value into `[-1, 1]` without centering (so it
preserves sparsity). `RobustScaler` centers on the median and divides by the
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

A constant column (zero variance, zero range, or zero IQR) is never divided by zero:
the scaler falls back to a scale of 1.0 (or maps to the bottom of `feature_range` for
`MinMaxScaler`), so the column survives the transform unchanged.

### Normalizing per row

`Normalizer` is the row-wise scaler: it divides each row by its norm across the named
columns, so every row becomes a unit vector. It is stateless (there is nothing to
learn), but still follows the `fit`/`transform` contract — use `transform` directly
after construction, or `fit_transform`. `norm="l2"` (default) divides by
`sqrt(Σ xᵢ²)`, `"l1"` by `Σ|xᵢ|`, and `"max"` by `max|xᵢ|`.

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

## Encoding categories

Categorical encoders learn the category set in `fit` (one `distinct` over the engine)
and lower `transform` to a `CASE` expression or a set of indicator columns — no
per-row Python.

`OrdinalEncoder` replaces each categorical column with an integer code in sorted
category order; `LabelEncoder` is the one-column variant for a target label.

```python
import batcher as bt
from batcher.ml.preprocessors import OrdinalEncoder, LabelEncoder

ds = bt.from_pydict({"city": ["paris", "rome", "paris", "oslo"]})

print(OrdinalEncoder(["city"]).fit_transform(ds).collect().column("city").to_pylist())
# [1, 2, 1, 0]
print(LabelEncoder("city").fit_transform(ds).collect().column("city").to_pylist())
# [1, 2, 1, 0]
```

`OneHotEncoder` drops each categorical column and adds one `{column}_{category}` 0/1
indicator per category (the scikit-learn naming convention). Pass `drop_first=True`
for dummy encoding (omit the first category to avoid collinearity).

```python
import batcher as bt
from batcher.ml.preprocessors import OneHotEncoder

ds = bt.from_pydict({"id": [1, 2, 3], "color": ["red", "green", "red"]})
encoded = OneHotEncoder(["color"]).fit_transform(ds).collect()
print(encoded.column_names)
# ['id', 'color_green', 'color_red']
print(encoded.to_pydict())
# {'id': [1, 2, 3], 'color_green': [0, 1, 0], 'color_red': [1, 0, 1]}
```

`MultiHotEncoder` is the multi-label counterpart for a **list** column (a tag set per
row): `fit` learns the distinct elements across all the lists, and `transform` emits
one indicator column per element, 1 where that element appears in the row's list. The
list column is kept alongside the indicators. Pass `categories=[...]` to fix the
vocabulary and skip `fit`.

```python
import batcher as bt
from batcher.ml.preprocessors import MultiHotEncoder

ds = bt.from_pydict({"tags": [["news", "sports"], ["news"], ["tech"]]})
encoded = MultiHotEncoder("tags").fit_transform(ds).collect()
print(encoded.column_names)
# ['tags', 'tags_news', 'tags_sports', 'tags_tech']
print(encoded.column("tags_news").to_pylist())
# [1, 1, 0]
```

`OrdinalEncoder`/`LabelEncoder` map unseen-at-fit values (and nulls) to
`unknown_value` (default `-1`); `OneHotEncoder` produces all-zero indicators for them.
That is why fit happens once on train: a category only present in validation still
encodes deterministically instead of shifting every code.

## Imputing missing values

`SimpleImputer` learns a per-column fill value in `fit` and replaces nulls with it in
`transform` (a `coalesce`, evaluated in the engine). `strategy` is `"mean"`,
`"median"`, `"most_frequent"`, or `"constant"` (which needs a `fill_value`). The
`"mean"`/`"median"` strategies cast the column to float (the scikit-learn
convention); `"most_frequent"`/`"constant"` keep the original type, so they also work
on string and categorical columns.

```python
import batcher as bt
from batcher.ml.preprocessors import SimpleImputer

train = bt.from_pydict({"age": [20.0, None, 40.0, None, 50.0]})
imputer = SimpleImputer(["age"], strategy="median").fit(train)
print(imputer.transform(train).collect().column("age").to_pylist())
# [20.0, 40.0, 40.0, 40.0, 50.0]
```

The learned fill value (`imputer.statistics_`) is reused on every split, so train and
validation get the *same* fill — the standard impute-then-scale ordering composes by
sequencing the objects, as the **Composing a pipeline** section below shows.

## Binning continuous values

`KBinsDiscretizer` turns a continuous column into an integer bin index `0..n_bins-1`.
`strategy="quantile"` (default) learns the quantile edges so each bin holds roughly
equal counts; `strategy="uniform"` learns equal-width edges from the min and max.

```python
import batcher as bt
from batcher.ml.preprocessors import KBinsDiscretizer

ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
binned = KBinsDiscretizer(["x"], n_bins=4, strategy="quantile").fit_transform(ds).collect()
print(binned.column("x").to_pylist())
# [0, 1, 1, 1, 2, 3, 3, 3]
```

## Assembling features

`Concatenator` stacks several numeric columns into one list column — the "make a
feature vector" step before training. It is stateless (`fit` is a no-op) but follows
the contract, so use `fit_transform` or `fit` then `transform`. The source columns
are kept unless `drop=True`.

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

The assembled list column becomes a tensor for training with zero or one copy (see
[PyTorch integration](pytorch.md)).

`Tokenizer` maps a text column through a user-supplied tokenizer (a `str -> list`
callable, or any object with `.encode`, such as a HuggingFace tokenizer). Tokenization
is inherently per-string, so it runs as a whole-batch `map_batches` UDF. It needs a
real tokenizer, so it is shown but not run here.

```python
# docs: skip
from batcher.ml.preprocessors import Tokenizer
from transformers import AutoTokenizer

hf = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenized = Tokenizer("text", hf, output_column="input_ids").fit_transform(ds)
```

## Composing a pipeline

A real feature pipeline is several preprocessors in sequence. Fit each on the previous
step's output, then push any split through the *same* fitted objects so train and
validation share every learned statistic. The classic order is impute → scale → encode.

```python
import batcher as bt
from batcher.ml.preprocessors import SimpleImputer, StandardScaler, OneHotEncoder

train = bt.from_pydict(
    {
        "age": [20.0, 30.0, None, 50.0],
        "income": [1.0, 2.0, 3.0, 4.0],
        "city": ["paris", "rome", "paris", "oslo"],
    }
)

imputer = SimpleImputer(["age"], strategy="median")
scaler = StandardScaler(["age", "income"])
encoder = OneHotEncoder(["city"])

# Fit each stage on the previous stage's output, on train only.
step1 = imputer.fit_transform(train)
step2 = scaler.fit_transform(step1)
prepared = encoder.fit_transform(step2)
print(prepared.collect().column_names)
# ['age', 'income', 'city_oslo', 'city_paris', 'city_rome']
```

Held-out data flows through the identical fitted objects — `transform`, never
`fit_transform`, so it inherits the training statistics:

```python
val = bt.from_pydict({"age": [None], "income": [2.5], "city": ["rome"]})
prepared_val = encoder.transform(scaler.transform(imputer.transform(val)))
print(prepared_val.collect().column_names)
# ['age', 'income', 'city_oslo', 'city_paris', 'city_rome']
```

## Where they run

`transform` is a lazy `Dataset`, so it composes with the rest of a pipeline and the
result is computed by a terminal op like `collect()` or `write.parquet(...)` — single
node or distributed. Use preprocessors before a training loop
([PyTorch integration](pytorch.md)) or before batch [inference](inference.md).
