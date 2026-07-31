# Encoding and imputation

This page covers turning non-numeric columns into numeric ones, filling the gaps, and
bucketing continuous values: the encoders, `SimpleImputer`, and `KBinsDiscretizer`.

A model needs numbers. Which encoder you want depends on the column's cardinality and
whether the categories have an order.

## Encoding categories

Categorical encoders learn the category set in `fit` with one `distinct` over the engine,
and lower `transform` to a `CASE` expression or a set of indicator columns. No per-row
Python runs anywhere in the path.

`OrdinalEncoder` replaces each categorical column with an integer code in sorted
category order. `LabelEncoder` is the one-column variant for a target label.

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
indicator per category, following the scikit-learn naming convention. Pass
`drop_first=True` for dummy encoding, which omits the first category to avoid
collinearity.

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

`MultiHotEncoder` is the multi-label counterpart for a **list** column holding a tag set
per row. `fit` learns the distinct elements across all the lists, and `transform` emits
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

`TargetEncoder` is the encoder for **high-cardinality** categoricals such as user IDs,
ZIP codes, and product SKUs, where one-hot would explode the width. It replaces each
category with a smoothed mean of a target column. That is the standard encoding for
gradient-boosted and linear tabular models, matching scikit-learn's `TargetEncoder`,
cuML, and `category_encoders`. `fit` is one mergeable `group_by(col).agg(count, sum)` per
column, so it scales to millions of categories across a cluster. The m-estimate smoothing
pulls rare categories toward the global mean, and unseen-at-fit categories map to that
prior. So **fit on the training split only**, or the target leaks into the features.

```python
import batcher as bt
from batcher.ml.preprocessors import TargetEncoder

ds = bt.from_pydict({"city": ["paris", "paris", "rome", "rome"], "churn": [1.0, 1.0, 0.0, 0.0]})
enc = TargetEncoder(["city"], "churn", smoothing=0.0).fit(ds)
print(enc.transform(ds).collect().column("city").to_pylist())
# [1.0, 1.0, 0.0, 0.0]  (paris churns, rome does not)
```

`OrdinalEncoder` and `LabelEncoder` map unseen-at-fit values, and nulls, to
`unknown_value`, which defaults to `-1`. `OneHotEncoder` produces all-zero indicators for
them. That is why fit happens once on train. A category only present in validation still
encodes deterministically instead of shifting every code.

## Encoding a high-cardinality category

`OneHotEncoder` and `OrdinalEncoder` both learn the category set, which caps them at the
cardinality the plan can express. Real categorical columns break that cap constantly: a URL
path, a product SKU, a user agent, a postcode. Three strategies handle it, in increasing
order of the cardinality they tolerate.

`FrequencyEncoder` replaces each category with how often it occurs. One numeric column, and
it often carries real signal, because a rare value behaves differently from a common one. An unseen
category encodes as 0, which is the correct answer.

```python
from batcher.ml.preprocessors import FrequencyEncoder

ds = bt.from_pydict({"agent": ["chrome", "chrome", "chrome", "curl"]})
print(FrequencyEncoder("agent").fit_transform(ds).to_pydict())
```

`RareCategoryEncoder` keeps the categories worth keeping and collapses the tail into one
bucket. This is the step that makes a one-hot encoding possible on a long-tailed column, and
it also fixes the serving-time unknown-category problem, because the bucket already exists.

```python
from batcher.ml.preprocessors import RareCategoryEncoder

ds = bt.from_pydict({"c": ["a"] * 90 + ["b"] * 9 + ["z"]})
encoder = RareCategoryEncoder("c", min_frequency=0.05).fit(ds)
print(encoder.transform(bt.from_pydict({"c": ["never_seen"]})).to_pydict())
```

`HashingEncoder` hashes into a fixed number of buckets. Unbounded cardinality, no fitted
state at all, and therefore no train/serve skew, at the cost of collisions, which a tree
model tolerates better than most people expect. It uses the engine's stable `xxhash64`
rather than Python's `hash()`, which varies per process and would be a silent skew.

`BinaryEncoder` is the middle ground between `OneHotEncoder` and `HashingEncoder` when a column has many categories but not unboundedly many: it assigns each category an integer and writes it in base 2, so 100 categories cost 7 bit columns rather than 100 one-hot columns, with no collisions. An unseen category encodes as all-zero bits.

## Weight-of-evidence encoding

`TargetEncoder` replaces a category with the target's mean; `WOEEncoder` replaces it with the
log-odds of the target relative to the overall odds. That is the transform a regulated credit
scorecard is built on, because WOE is additive in the log-odds space a logistic regression
works in. A WOE-encoded feature enters a linear model as a straight, interpretable
coefficient.

```python
import batcher as bt
from batcher.ml.preprocessors import WOEEncoder

ds = bt.from_pydict({"grade": ["a", "a", "b", "b"], "default": [0, 0, 1, 1]})
encoded = WOEEncoder(["grade"], "default").fit_transform(ds).to_pydict()["grade"]
print(encoded[0] < 0 < encoded[2])   # grade a leans safe, grade b leans default
```

Like `TargetEncoder` it is supervised, so fit it on the training split only. An unseen or
single-class category encodes as a neutral 0 rather than an infinite log-odds.

## Imputing missing values

`SimpleImputer` learns a per-column fill value in `fit` and replaces nulls with it in
`transform`, using a `coalesce` evaluated in the engine. `strategy` is `"mean"`,
`"median"`, `"most_frequent"`, or `"constant"`, and `"constant"` needs a `fill_value`. The
`"mean"` and `"median"` strategies cast the column to float, following the scikit-learn
convention. `"most_frequent"` and `"constant"` keep the original type, so they also work
on string and categorical columns.

```python
import batcher as bt
from batcher.ml.preprocessors import SimpleImputer

train = bt.from_pydict({"age": [20.0, None, 40.0, None, 50.0]})
imputer = SimpleImputer(["age"], strategy="median").fit(train)
print(imputer.transform(train).collect().column("age").to_pylist())
# [20.0, 40.0, 40.0, 40.0, 50.0]
```

The learned fill value in `imputer.statistics_` is reused on every split, so train and
validation get the *same* fill. The standard impute-then-scale ordering composes by
sequencing the objects, as **Composing a pipeline** below shows.

## Binning continuous values

`KBinsDiscretizer` turns a continuous column into an integer bin index `0..n_bins-1`.
The default `strategy="quantile"` learns the quantile edges so each bin holds roughly
equal counts. `strategy="uniform"` learns equal-width edges from the min and max.

```python
import batcher as bt
from batcher.ml.preprocessors import KBinsDiscretizer

ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
binned = KBinsDiscretizer(["x"], n_bins=4, strategy="quantile").fit_transform(ds).collect()
print(binned.column("x").to_pylist())
# [0, 1, 1, 1, 2, 3, 3, 3]
```

When the edges are known up front rather than learned from the data, `bt.cut` is the pure expression for the job. It needs no `fit`, takes explicit break points, and returns the integer bin index or a label per bucket, so it composes anywhere an expression does and runs in one streaming pass.

```python
ds = bt.from_pydict({"age": [5, 18, 40, 70]})
banded = ds.with_columns(band=bt.cut("age", [12, 19, 65], labels=["child", "teen", "adult", "senior"]))
print(banded.to_pydict()["band"])
# ['child', 'teen', 'adult', 'senior']
```

## See also

- {doc}`/ml/preparing/preprocessors/scaling`: the numeric half of the same job.
- {doc}`/ml/preparing/preprocessors/feature-generation`: deriving new columns once the existing ones are numeric.
- {doc}`/ml/preparing/preprocessors/index`: the fit/transform contract and the full preprocessor table.
