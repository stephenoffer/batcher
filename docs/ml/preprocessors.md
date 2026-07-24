# Preprocessors

Preprocessors are scikit-learn-style `fit` and `transform` feature transformers that run
on the engine. `fit` learns its state with one mergeable aggregate over the data, so it
is distributed and spillable for free. `transform` is a lazy column rewrite. Fit
on the training set, then `transform` the training **and** validation sets with the
same learned state.

Every preprocessor is importable from `batcher.ml.preprocessors`. Most are also
re-exported from `batcher.ml`, with the exceptions of `TargetEncoder` and
`PolynomialFeatures`, which you import from `batcher.ml.preprocessors`.

## Splitting first

`fit` must see the training rows only. Fit on the whole frame and held-out statistics
leak into your features. `ds.ml.train_test_split` gives disjoint parts that
together cover every row, assigned by a reproducible hash of each row's own content.
Each part is a plain row-wise filter, so the split streams, distributes, and is
*partition-independent*. A row lands in the same part however the data is laid out.

```python
import batcher as bt

ds = bt.range(0, 1000)
train, test = ds.ml.train_test_split(0.2, seed=42)
print(train.count() + test.count())
# 1000
```

`ds.ml.random_split([0.7, 0.15, 0.15], seed=42)` generalizes it to a
train/validation/test split.

Pass `key=` to hash only the columns that identify a row:

```python
users = bt.range(0, 1000).select(id=bt.col("value"), score=bt.col("value") * 2)
train, test = users.ml.train_test_split(0.2, seed=42, key="id")
print(train.count() + test.count())
# 1000
```

Then re-deriving a feature column does not move rows between train and test. The split
follows `id` alone, so recomputing `score` leaves every row where it was:

```python
rescored = users.with_columns(score=bt.col("id") * 3)
again, _ = rescored.ml.train_test_split(0.2, seed=42, key="id")
print(sorted(again.to_pydict()["id"]) == sorted(train.to_pydict()["id"]))
# True
```

Without `key=` every column is hashed, which is correct but re-splits whenever any value
changes.

## Fuzzy deduplication

Exact deduplication is `distinct()`. On a web-scale training corpus it barely helps,
because the duplicates are the same article behind a different header, or the same page
with a changed timestamp. Removing *those* is the single biggest win in preprocessing an
LLM pretraining set.

`ds.ml.near_duplicates` finds the pairs, and `ds.ml.drop_near_duplicates` removes them,
keeping one representative per cluster.

```python
import batcher as bt

docs = bt.from_pydict({"text": [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox jumps over the lazy dog!",   # near-duplicate
    "a treatise on the migratory habits of geese",
]})
print(docs.ml.drop_near_duplicates("text", threshold=0.7).count())
# 2
print(docs.distinct().count())   # exact dedup keeps all three
# 3
```

Under the hood, `str.minhash` reduces each document to a fixed-length signature whose
positional agreement rate, computed by `list.jaccard`, estimates the documents' Jaccard
similarity. LSH banding then turns the similarity join into an equi-join on a band hash.
Every returned pair is **verified** against the threshold, so banding only costs recall,
never precision. `bands` is the dial. More bands means more candidates, more recall, and
more work.

Both are ordinary relational plans built from a projection, an `explode`, and some joins,
so they run wherever a join runs.

## Matching on meaning with similarity_join

MinHash answers "are these two documents made of the same words". It says nothing about
two rows that *mean* the same thing in different words. That is a question for embeddings,
and `ds.ml.similarity_join` is the same two-stage recipe with the signature swapped:
`.list.simhash` replaces `str.minhash`, and the verification is the **exact**
`list.cosine_similarity` over the original vectors.

```python
import batcher as bt

catalog = bt.from_pydict({"sku": [1, 2], "v": [[1.0, 0.0], [0.0, 1.0]]})
feed = bt.from_pydict({"ref": [10], "v": [[1.0, 0.02]]})
pairs = catalog.ml.similarity_join(
    feed, left_on="v", threshold=0.9, left_key="sku", right_key="ref"
)
print(pairs.select("key_a", "key_b").to_pydict())
# {'key_a': [1], 'key_b': [10]}
```

This is entity resolution, matching a product catalog against a supplier feed or a CRM
against a billing system, and it is also retrieval over a corpus. It covers any join
whose key is "means the same thing" rather than "is the same string".

`simhash` is Charikar's random-hyperplane LSH. `num_bits` hyperplanes are drawn through
the origin and each bit records which side of one the vector falls on. Two vectors an
angle `theta` apart agree on each bit with probability `1 - theta/pi`, so the fraction of
agreeing bits estimates the angle. That is the vector-space counterpart of MinHash's
Jaccard estimate. The hyperplanes are derived by hashing `(seed, bit, dimension)` rather
than stored, so every partition and every machine draws the same ones and a signature
computed on one node is comparable with one computed on another.

Exactly as in fuzzy dedup, banding governs **recall, never precision**. No pair below
`threshold` is ever returned, but a pair above it can miss every band. `bands` is the
dial. Rows whose vector is null or empty have no direction, cannot clear any threshold,
and are dropped rather than banded. Left in, they would all collide and blow the
candidate set up quadratically.

## Chaining steps

`Chain` is the sklearn `Pipeline` equivalent. It fits each step on the **previous
step's output** and replays the fitted steps, in order, over any split. Doing this by
hand means fitting step *i* on data that steps *0..i-1* have already transformed. That
is easy to get subtly wrong, and the mistake leaks held-out statistics into training
features without ever failing.

```python
import batcher as bt
from batcher.ml import Chain, SimpleImputer, StandardScaler

ds = bt.from_pydict({"age": [10.0, 20.0, None, 40.0, 30.0, 50.0]})
train, test = ds.ml.train_test_split(0.3, seed=0)

chain = Chain(SimpleImputer(["age"]), StandardScaler(["age"])).fit(train)
train_x, test_x = chain.transform(train), chain.transform(test)
print(chain)
# Chain(SimpleImputer, StandardScaler)
```

Call `fit` on the training split only, then `transform` on both. A `Chain` is itself a
`Preprocessor`, so it nests. Its steps stay introspectable through `chain[0]` and
`len(chain)`, which is how you read a fitted step's learned state.

You can also sequence several preprocessors by hand. Fit each on the previous
step's output, then transform any split through the same fitted objects.

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

Each object keeps its fitted state, so the same steps transform held-out data with the
statistics learned on train:

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

Every preprocessor exposes the same `Preprocessor` API. `fit(ds)` runs a small aggregate,
stores the learned state on the object, and returns `self`. Even a stateless transform
such as `Normalizer`, `Concatenator`, or `Tokenizer` needs a `fit` or `fit_transform`
before `transform`. `transform(ds)` returns a new lazy `Dataset` with the learned rewrite
applied, and runs no work until a terminal op such as `collect` or `write.parquet`.
`fit_transform(ds)` is `fit(ds).transform(ds)`, the common single-split path.

`fit` is the one place a preprocessor *executes* and touches data. `transform` stays
lazy, so it composes with the rest of the pipeline and runs inside the engine. Calling
`transform` before `fit` raises `PlanError`.

## Available preprocessors

The table lists every preprocessor, what its `fit` learns, and what its `transform`
does. Stateless entries learn nothing and only need a `fit` call to satisfy the contract.

| Class | `fit` learns | `transform` |
| --- | --- | --- |
| `StandardScaler` | mean, population std | `(x - mean) / std` |
| `MinMaxScaler` | min, max | scale into `feature_range`, default `[0, 1]` |
| `MaxAbsScaler` | max absolute value | `x / max(\|x\|)` into `[-1, 1]` |
| `RobustScaler` | median, IQR | `(x - median) / IQR`, outlier-robust |
| `OrdinalEncoder` | sorted categories | integer code per category |
| `LabelEncoder` | sorted classes | integer code for one target column |
| `OneHotEncoder` | categories | one 0/1 indicator column per category |
| `BinaryEncoder` | categories | the category's integer code in base 2, one column per bit |
| `MultiHotEncoder` | distinct list elements | one 0/1 indicator column per category, for a list column |
| `TargetEncoder` | per-category target mean, global prior | smoothed mean-target code per high-cardinality category |
| `KBinsDiscretizer` | bin edges, quantile or uniform | integer bin index `0..n_bins-1` |
| `Normalizer` | nothing, stateless | scale each row to unit L1, L2, or max norm across columns |
| `SimpleImputer` | mean, median, mode, or constant | fill nulls |
| `Concatenator` | nothing, stateless | stack columns into one tensor column |
| `PolynomialFeatures` | nothing, stateless | add interaction and power terms such as `a*b` and `a^2` up to a degree |
| `Tokenizer` | nothing, stateless | tokenize text with a user tokenizer |
| `QuantileTransformer` | `n_quantiles` cut points | map onto a uniform or normal distribution by rank |
| `PowerTransformer` | the Yeo-Johnson lambda, by maximum likelihood | make a skewed column more Gaussian |
| `BoxCoxTransformer` | the Box-Cox lambda, by maximum likelihood | the same, for a strictly positive column |
| `LogTransformer` | nothing, stateless | `log(x + offset)`, the explainable shape fix |
| `Clipper` | lower and upper quantiles | clamp values into the learned range |
| `MissingIndicator` | nothing, stateless | append a boolean flag per column, before imputation |
| `FrequencyEncoder` | per-category frequency | replace a category with how often it occurs |
| `RareCategoryEncoder` | the categories clearing `min_frequency` | collapse the tail into one bucket |
| `HashingEncoder` | nothing, stateless | hash a category into one of `n_buckets` |

All preprocessors share the `Preprocessor` base contract of `fit`, `transform`, and
`fit_transform`.

Each scaler matches scikit-learn's definitions, and `StandardScaler` uses population
variance. `fit` lowers to the existing `group_by().agg(...)` and `distinct()`
operators, so it is partition-independent. A fit on a distributed dataset learns the
same statistics as a single-node fit.

## Rank and label transforms

`RankTransformer` replaces a value with its percentile rank — like `QuantileTransformer` it
keeps only the order and is immune to outliers, but it is exact (every distinct value gets its
own rank) rather than binned, which matters on a small column.

```python
import batcher as bt
from batcher.ml.preprocessors import RankTransformer

ds = bt.from_pydict({"x": [10.0, 40.0, 20.0, 1000.0]})
print(RankTransformer("x").fit_transform(ds).to_pydict()["x"])
```

`LabelBinarizer` one-vs-rest expands a categorical *label* into a 0/1 column per class — the
target-side counterpart of one-hot encoding, for a per-class metric or a set of binary models.
`MultiLabelBinarizer` does the same for a *list* column, where a row can carry many labels at
once (tags, genres), which is the standard input shaping for a multi-label classifier.

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
makes the column most Gaussian by maximum likelihood — and does it in **one pass**, because
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
Missingness is usually a signal — a blank income field means something different from a low
one — and imputing first destroys it permanently.

```python
from batcher.ml.preprocessors import Chain, MissingIndicator, SimpleImputer

ds = bt.from_pydict({"income": [50000.0, None, 70000.0]})
flagged = Chain(MissingIndicator("income"), SimpleImputer(["income"])).fit_transform(ds)
print(flagged.to_pydict()["income_missing"])
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

## Encoding a high-cardinality category

`OneHotEncoder` and `OrdinalEncoder` both learn the category set, which caps them at the
cardinality the plan can express. Real categorical columns break that cap constantly: a URL
path, a product SKU, a user agent, a postcode. Three strategies handle it, in increasing
order of the cardinality they tolerate.

`FrequencyEncoder` replaces each category with how often it occurs. One numeric column, and
it often carries real signal — a rare value behaves differently from a common one. An unseen
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
state at all, and therefore no train/serve skew — at the cost of collisions, which a tree
model tolerates better than most people expect. It uses the engine's stable `xxhash64`
rather than Python's `hash()`, which varies per process and would be a silent skew.

`BinaryEncoder` is the middle ground between `OneHotEncoder` and `HashingEncoder` when a column has many categories but not unboundedly many: it assigns each category an integer and writes it in base 2, so 100 categories cost 7 bit columns rather than 100 one-hot columns, with no collisions. An unseen category encodes as all-zero bits.

## Weight-of-evidence encoding

`TargetEncoder` replaces a category with the target's mean; `WOEEncoder` replaces it with the
log-odds of the target relative to the overall odds. That is the transform a regulated credit
scorecard is built on, because WOE is additive in the log-odds space a logistic regression
works in — a WOE-encoded feature enters a linear model as a straight, interpretable
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

## Saving a fitted preprocessor

A preprocessor is only useful because its state is learned once and reused: the scaler
standardizing a request at serving time must hold the *training* set's mean. `save` writes
that state as plain JSON.

```python
import os
import tempfile

from batcher.ml.preprocessors import Preprocessor, StandardScaler

scaler = StandardScaler("x").fit(bt.from_pydict({"x": [1.0, 3.0]}))
path = os.path.join(tempfile.mkdtemp(), "scaler.json")
scaler.save(path)
print(Preprocessor.load(path).mean_)
```

JSON rather than a pickle, deliberately: the file is reviewable, diffable, portable to a
serving stack in another language, and safe to load from a store you do not fully control.
A cloud URI works wherever a local path does.

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
[PyTorch integration](pytorch.md).

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

## Composing a pipeline

A real feature pipeline is several preprocessors in sequence. Fit each on the previous
step's output, then push any split through the *same* fitted objects so train and
validation share every learned statistic. The classic order is impute, then scale, then
encode.

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

Held-out data flows through the identical fitted objects. Use `transform`, never
`fit_transform`, so it inherits the training statistics:

```python
val = bt.from_pydict({"age": [None], "income": [2.5], "city": ["rome"]})
prepared_val = encoder.transform(scaler.transform(imputer.transform(val)))
print(prepared_val.collect().column_names)
# ['age', 'income', 'city_oslo', 'city_paris', 'city_rome']
```

## Where they run

`transform` is a lazy `Dataset`, so it composes with the rest of a pipeline and the
result is computed by a terminal op such as `collect()` or `write.parquet(...)`, on one
node or across a cluster. Use preprocessors before a training loop, covered in
[PyTorch integration](pytorch.md), or before batch [inference](inference.md).

## Next steps

- [Feature engineering tutorial](../tutorials/feature-engineering.md): the full workflow
  from raw table to model-ready matrix, end to end, with `Chain`.
- [PyTorch integration](pytorch.md): hand the assembled features to a training loop.
- [ML API reference](../api/ml.md): the complete `Preprocessor` surface.
