# Preprocessors

Preprocessors are scikit-learn-style `fit`/`transform` feature transformers that run
on the engine. `fit` learns its state with one mergeable aggregate over the data (so
it is distributed and spillable for free); `transform` is a lazy column rewrite. Fit
on the training set, then `transform` the training **and** validation sets with the
same learned state.

Every preprocessor is importable from `batcher.ml` as well as
`batcher.ml.preprocessors`.

## Splitting first

`fit` must see the training rows only. Fit on the whole frame and held-out statistics
leak into your features. `ds.ml.train_test_split` gives disjoint parts that
together cover every row, assigned by a reproducible hash of each row's own content.
Each part is a plain row-wise filter, so the split streams, distributes, and is
*partition-independent*: a row lands in the same part however the data is laid out.

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

Then re-deriving a feature column does not move rows between train and test: the split
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

Exact deduplication is `distinct()`. On a web-scale training corpus it barely helps: the
duplicates are the same article behind a different header, or the same page with a
changed timestamp. Removing *those* is the single biggest win in preprocessing an LLM
pretraining set.

`ds.ml.near_duplicates` finds the pairs; `ds.ml.drop_near_duplicates` removes them,
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

Under the hood: `str.minhash` reduces each document to a fixed-length signature whose
positional agreement rate (`list.jaccard`) estimates the documents' Jaccard similarity,
and LSH banding turns the similarity join into an equi-join on a band hash. Every
returned pair is then **verified** against the threshold, so banding only costs recall,
never precision. `bands` is the dial: more bands, more candidates, more recall, more work.

Both are ordinary relational plans (a projection, an `explode`, some joins), so they run
wherever a join runs.

## Matching on meaning: `similarity_join`

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

This is entity resolution (a product catalog against a supplier feed, a CRM against a
billing system) and retrieval over a corpus: any join whose key is "means the same
thing" rather than "is the same string".

`simhash` is Charikar's random-hyperplane LSH: `num_bits` hyperplanes are drawn through
the origin and each bit records which side of one the vector falls on. Two vectors an
angle `θ` apart agree on each bit with probability `1 - θ/π`, so the fraction of agreeing
bits estimates the angle: the vector-space counterpart of MinHash's Jaccard estimate.
The hyperplanes are derived by hashing `(seed, bit, dimension)` rather than stored, so
every partition and every machine draws the same ones and a signature computed on one
node is comparable with one computed on another.

Exactly as in fuzzy dedup, banding governs **recall, never precision**: no pair below
`threshold` is ever returned, but a pair above it can miss every band. `bands` is the
dial. Rows whose vector is null or empty have no direction, cannot clear any threshold,
and are dropped rather than banded. Left in, they would all collide and blow the
candidate set up quadratically.

## Chaining steps

`Chain` is the sklearn `Pipeline` equivalent: it fits each step on the **previous
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

`fit` on the training split only; `transform` both. A `Chain` is itself a
`Preprocessor`, so it nests. Its steps stay introspectable (`chain[0]`, `len(chain)`)
to read a fitted step's learned state.

Or compose several preprocessors by sequencing them by hand: fit each on the previous
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
stores the learned state on the object, and returns `self`; even a stateless transform
(`Normalizer`, `Concatenator`, `Tokenizer`) needs a `fit` or `fit_transform` before
`transform`. `transform(ds)` returns a new lazy `Dataset` with the learned rewrite
applied, and runs no work until a terminal op (`collect`, `write.parquet`, ...).
`fit_transform(ds)` is `fit(ds).transform(ds)`, the common single-split path.

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
held-out split with the training mean and standard deviation. Never refit on
validation data, or the splits no longer share a scale:

```python
val = bt.from_pydict({"age": [35.0], "score": [2.5]})
print(scaler.transform(val).collect().column("age").to_pylist())
# [0.0] — 35.0 is the training mean, so it standardizes to zero
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
learn), but still follows the `fit`/`transform` contract, so use `transform` directly
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
and lower `transform` to a `CASE` expression or a set of indicator columns. No per-row
Python anywhere in the path.

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
validation get the *same* fill. The standard impute-then-scale ordering composes by
sequencing the objects, as **Composing a pipeline** below shows.

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

`Concatenator` stacks several numeric columns into one list column: the "make a
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
result is computed by a terminal op like `collect()` or `write.parquet(...)`, on one
node or across a cluster. Use preprocessors before a training loop
([PyTorch integration](pytorch.md)) or before batch [inference](inference.md).

## Next steps

- [Feature engineering tutorial](../tutorials/feature-engineering.md): the full raw
  table → model-ready matrix workflow, end to end, with `Chain`.
- [PyTorch integration](pytorch.md): hand the assembled features to a training loop.
- [ML API reference](../api/ml.md): the complete `Preprocessor` surface.
