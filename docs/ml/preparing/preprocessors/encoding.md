# Encoding and imputation

This page covers turning non-numeric columns into numeric ones, filling the gaps, and
bucketing continuous values: the encoders, {py:class}`SimpleImputer <batcher.ml.preprocessors.SimpleImputer>`, and {py:class}`KBinsDiscretizer <batcher.ml.preprocessors.KBinsDiscretizer>`.
Which encoder you want depends on two things: how many categories the column has, and
whether they have an order.

## Encoding categories

Categorical encoders learn the category set in `fit` with one `distinct` over the engine,
and lower `transform` to a `CASE` expression or a set of indicator columns. No per-row
Python runs anywhere.

{py:class}`OrdinalEncoder <batcher.ml.preprocessors.OrdinalEncoder>` replaces each categorical column with an integer code in sorted
category order. {py:class}`LabelEncoder <batcher.ml.preprocessors.LabelEncoder>` is the one-column variant for a target label.

```python
import batcher as bt
from batcher.ml.preprocessors import OrdinalEncoder, LabelEncoder

ds = bt.from_pydict({"city": ["paris", "rome", "paris", "oslo"]})

print(OrdinalEncoder(["city"]).fit_transform(ds).collect().column("city").to_pylist())
# [1, 2, 1, 0]
print(LabelEncoder("city").fit_transform(ds).collect().column("city").to_pylist())
# [1, 2, 1, 0]
```

{py:class}`OneHotEncoder <batcher.ml.preprocessors.OneHotEncoder>` drops each categorical column and adds one `{column}_{category}` 0/1
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

{py:class}`MultiHotEncoder <batcher.ml.preprocessors.MultiHotEncoder>` is the multi-label counterpart for a list column holding a tag set
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

{py:class}`TargetEncoder <batcher.ml.preprocessors.TargetEncoder>` is the encoder for high-cardinality categoricals such as user IDs,
ZIP codes, and product SKUs, where one-hot would explode the width. It replaces each
category with a smoothed mean of a target column, the standard encoding for
gradient-boosted and linear tabular models, as in scikit-learn's `TargetEncoder`, cuML, and
`category_encoders`. `fit` is one mergeable {py:meth}`group_by(col).agg(count, sum) <batcher.Dataset.group_by>` per
column, so it distributes like any aggregate. The m-estimate `smoothing` pulls rare
categories toward the global mean, and a category unseen at fit maps to that prior.

The encoding reads the target, so fit it on the training split only. Otherwise the target
leaks into the features. Pass `cv=k` and `fit_transform` returns a cross-fitted encoding
instead, where each training row is encoded from the other `k-1` folds only.

```python
import batcher as bt
from batcher.ml.preprocessors import TargetEncoder

ds = bt.from_pydict({"city": ["paris", "paris", "rome", "rome"], "churn": [1.0, 1.0, 0.0, 0.0]})
enc = TargetEncoder(["city"], "churn", smoothing=0.0).fit(ds)
print(enc.transform(ds).collect().column("city").to_pylist())
# [1.0, 1.0, 0.0, 0.0]  (paris churns, rome does not)
```

`OrdinalEncoder` and `LabelEncoder` map values unseen at fit, and nulls, to
`unknown_value`, which defaults to `-1`. `OneHotEncoder` produces all-zero indicators for
them. A category present only in validation therefore encodes deterministically instead of
shifting every code, which is why you fit once, on train.

## Encoding a high-cardinality category

`OneHotEncoder` and `OrdinalEncoder` turn every learned category into a `CASE` arm or an
output column, so the category set is capped by `max_categories`, 1,000 by default. Past
that, `fit` raises rather than building a plan with a million arms. Real columns cross that
line all the time: a URL path, a product SKU, a user agent, a postcode. The encoders below
handle them.

{py:class}`FrequencyEncoder <batcher.ml.preprocessors.FrequencyEncoder>` replaces each category with its share of rows, or its raw count with
`normalize=False`. That's one numeric column, and it often carries real signal, because a
rare value behaves differently from a common one. An unseen category encodes as 0, its
training frequency, and so does a null.

```python
from batcher.ml.preprocessors import FrequencyEncoder

ds = bt.from_pydict({"agent": ["chrome", "chrome", "chrome", "curl"]})
print(FrequencyEncoder("agent").fit_transform(ds).to_pydict())
```

{py:class}`RareCategoryEncoder <batcher.ml.preprocessors.RareCategoryEncoder>` keeps each category above `min_frequency` and collapses the tail into one
bucket. That makes a one-hot encoding possible on a long-tailed column. It also settles the
serving-time unknown category, because the bucket already exists.

```python
from batcher.ml.preprocessors import RareCategoryEncoder

ds = bt.from_pydict({"c": ["a"] * 90 + ["b"] * 9 + ["z"]})
encoder = RareCategoryEncoder("c", min_frequency=0.05).fit(ds)
print(encoder.transform(bt.from_pydict({"c": ["never_seen"]})).to_pydict())
```

{py:class}`HashingEncoder <batcher.ml.preprocessors.HashingEncoder>` hashes into `n_buckets` buckets. Cardinality is unbounded and there's no
fitted state, so there's no train/serve skew either. The price is collisions. It hashes with
the engine's stable `str.xxhash64()`, not Python's `hash()`, which varies from process to
process and would skew serving silently.

{py:class}`BinaryEncoder <batcher.ml.preprocessors.BinaryEncoder>` sits between `OneHotEncoder` and `HashingEncoder`. It assigns each category an integer and writes it in base 2, so 100 categories cost 7 bit columns instead of 100, with no collisions. An unseen category encodes as all-zero bits. It still learns the category set, so `max_categories` applies.

## Choosing how much to trust a category

`TargetEncoder` shrinks every category toward the global mean by the same fixed weight. Two
other encoders decide differently, and the difference shows on a long-tailed column where
categories differ wildly in size.

{py:class}`LeaveOneOutEncoder <batcher.ml.preprocessors.LeaveOneOutEncoder>` doesn't shrink
at all. It removes the row's own contribution instead, so a row's encoding is the mean of
the *other* rows in its category and the target cannot leak into its own feature:

```python
from batcher.ml.preprocessors import LeaveOneOutEncoder

spend = bt.from_pydict({"city": ["a", "a", "a"], "amount": [0.0, 3.0, 6.0]})
print(LeaveOneOutEncoder(["city"], "amount").fit_transform(spend).to_pydict()["city"])
# [4.5, 3.0, 1.5]
```

`fit_transform` applies that leave-one-out form because those are the training rows;
`transform` applies the plain category mean, because a held-out row contributed nothing to
subtract.

That exactness has a cost. On a binary target in a category of two rows, the encoding *is*
the other row's label, so a model can learn to read the feature backwards. Use it where
categories have a reasonable number of rows each, and `TargetEncoder(cv=...)` when the tail
is thin.

{py:class}`JamesSteinEncoder <batcher.ml.preprocessors.JamesSteinEncoder>` derives the
shrinkage from the data rather than taking it as a hyperparameter. A category whose own
target scatters widely relative to the spread between categories is trusted less; one that
is both large and consistent keeps almost all of its own mean:

```python
from batcher.ml.preprocessors import JamesSteinEncoder

mixed = bt.from_pydict({"city": ["big"] * 20 + ["tiny"] * 2, "churn": [1.0] * 20 + [1.0, 1.0]})
fitted = JamesSteinEncoder(["city"], "churn").fit(mixed)
print(round(fitted.mapping_["city"]["big"], 4) >= round(fitted.mapping_["city"]["tiny"], 4))
# True
```

That removes the one number `TargetEncoder` asks you to guess. A single `smoothing` that
suits a category with ten rows over-shrinks one with ten thousand.

Both learn from one mergeable `group_by` per column and encode with a lazy `CASE`
expression, like every other encoder here. An unseen category, and a null, take the global mean.

## Weight-of-evidence encoding

{py:class}`WOEEncoder <batcher.ml.preprocessors.WOEEncoder>` replaces a category with the log-odds of the target relative to the
overall odds, where `TargetEncoder` uses the target's mean. Credit scorecards are built on
it, because weight of evidence is additive in the log-odds space a logistic regression works
in. A WOE-encoded feature enters a linear model as one straight, interpretable
coefficient.

```python
import batcher as bt
from batcher.ml.preprocessors import WOEEncoder

ds = bt.from_pydict({"grade": ["a", "a", "b", "b"], "default": [0, 0, 1, 1]})
encoded = WOEEncoder(["grade"], "default").fit_transform(ds).to_pydict()["grade"]
print(encoded[0] < 0 < encoded[2])  # grade a leans safe, grade b leans default
```

Like `TargetEncoder` it is supervised, so fit it on the training split only. An unseen or
single-class category encodes as a neutral 0 rather than an infinite log-odds.

## Imputing missing values

`SimpleImputer` learns a per-column fill value in `fit` and replaces nulls with it in
`transform`, using a `coalesce` evaluated in the engine. `strategy` is `"mean"`,
`"median"`, `"most_frequent"`, or `"constant"`, and `"constant"` needs a `fill_value`. The
`"mean"` and `"median"` strategies cast the column to float, following the scikit-learn
convention. `"most_frequent"` and `"constant"` keep the original type, so they also work
on string and categorical columns. When two values are equally frequent, `"most_frequent"`
picks the smallest, as scikit-learn does, so the fill value doesn't depend on how the data
was partitioned. In a float column, NaN is missing too: it is skipped when the fill value is
learned and filled by `transform`.

```python
import batcher as bt
from batcher.ml.preprocessors import SimpleImputer

train = bt.from_pydict({"age": [20.0, None, 40.0, None, 50.0]})
imputer = SimpleImputer(["age"], strategy="median").fit(train)
print(imputer.transform(train).collect().column("age").to_pylist())
# [20.0, 40.0, 40.0, 40.0, 50.0]
```

The learned fill value in `imputer.statistics_` is reused on every split, so train and
validation get the same fill. Impute, then scale. {doc}`/ml/preparing/preprocessors/pipelines`
covers chaining the two so the order can't slip.

## Imputing from the other columns

{py:class}`SimpleImputer <batcher.ml.preprocessors.SimpleImputer>` fills a column with one
number, which discards everything the row's *other* columns say about it. A missing income
in a row with a known job title and postcode is not well described by the global mean.

{py:class}`IterativeImputer <batcher.ml.preprocessors.IterativeImputer>` is the MICE-style
alternative, as in scikit-learn. It models each incomplete column from the remaining ones,
fills the gaps with the model's prediction, and repeats, so later rounds see better fills
than earlier ones did.

```python
from batcher.ml.preprocessors import IterativeImputer, SimpleImputer

related = bt.from_pydict(
    {"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "b": [2.0, 4.0, None, 8.0, 10.0, 12.0]}
)
print(round(IterativeImputer(["a", "b"]).fit_transform(related).to_pydict()["b"][2], 3))
# 6.0
print(round(SimpleImputer(["b"]).fit_transform(related).to_pydict()["b"][2], 3))
# 7.2
```

`b` is exactly twice `a`, so 6.0 is the right answer. 7.2 is the column mean.

`fit` records the entire schedule: the initial per-column fill, then one model per
incomplete column per round, in order. `transform` replays it. A serving row is therefore
imputed by the same models in the same sequence as a training row, which is the part a
hand-rolled loop usually gets wrong.

The cost is real. A fit is up to `max_iter` (10 by default) times the number of incomplete
columns in model fits, each a pass over the data, so use `SimpleImputer` when the columns
are unrelated. Rounds stop early once nothing moves by more than `tol`, and `n_iter_`
reports how many ran.

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

When you know the edges up front, use {py:func}`bt.cut <batcher.cut>` instead. It's a plain expression with no `fit`: it takes explicit break points and returns the integer bin index, or a label per bucket, so it composes anywhere an expression does.

```python
ds = bt.from_pydict({"age": [5, 18, 40, 70]})
banded = ds.with_columns(
    band=bt.cut("age", [12, 19, 65], labels=["child", "teen", "adult", "senior"])
)
print(banded.to_pydict()["band"])
# ['child', 'teen', 'adult', 'senior']
```

## See also

- {doc}`/ml/preparing/preprocessors/scaling`: the numeric half of the same job.
- {doc}`/ml/preparing/preprocessors/feature-generation`: deriving new columns once the existing ones are numeric.
- {doc}`/ml/preparing/preprocessors/text-vectorization`: turning a free-text column into features, rather than a categorical one.
- {doc}`/ml/preparing/preprocessors/index`: the fit/transform contract and the full preprocessor table.
