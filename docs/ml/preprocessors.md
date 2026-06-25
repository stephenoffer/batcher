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
```

Because the fitted state lives on each object, the same steps transform held-out
data with the statistics learned on train:

```python
import batcher as bt
from batcher.ml.preprocessors import StandardScaler

train = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
scaler = StandardScaler(["x"]).fit(train)

test = bt.from_pydict({"x": [6.0, 7.0]})
print(scaler.transform(test).collect().column("x").to_pylist())
```

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

## Encoding categories

```python
import batcher as bt
from batcher.ml.preprocessors import OneHotEncoder

ds = bt.from_pydict({"id": [1, 2, 3], "color": ["red", "green", "red"]})
encoded = OneHotEncoder(["color"]).fit_transform(ds).collect()
print(encoded.column_names)  # id, color_green, color_red
```

`OrdinalEncoder`/`LabelEncoder` map unseen-at-fit values (and nulls) to
`unknown_value` (default `-1`); `OneHotEncoder` produces all-zero indicators for them.

## Where they run

`transform` is a lazy `Dataset`, so it composes with the rest of a pipeline and the
result is computed by a terminal op like `collect()` or `write.parquet(...)` — single
node or distributed. Use preprocessors before a training loop
([PyTorch integration](pytorch.md)) or before batch [inference](inference.md).
