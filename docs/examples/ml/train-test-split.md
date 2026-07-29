# Train/test split

Every leaked test score traces back to a split. The model looks excellent offline and
mediocre in production, and the reason is almost never the architecture. It is that a row in
the test set had a twin in the training set. This page covers the three ways that
happens and what to do instead.

:::{warning}
None of the three leaks below raises anything. The split succeeds, the training run succeeds,
the metric comes out higher than it should, and the first thing that disagrees with you is
production.
:::

| The leak | What it looks like | The fix |
| --- | --- | --- |
| The same entity on both sides | a user's events split across train and test; the model memorizes the user | `key="user_id"`, which hashes the entity rather than the row |
| The split moves when you touch a feature | two experiments compared on two different test sets | name a stable `key`; the default hashes every column |
| The future in the training set | a random split over time-ordered data trains on Tuesday to predict Monday | cut on time with a `filter`, and hold out a gap |

`ds.ml.train_test_split` assigns each row by a reproducible hash of its own values. The
parts are disjoint, they cover every row, and the assignment does not depend on how the
data is partitioned: single-node, multi-core, distributed, or streaming all produce the
same split. Nothing is shuffled and nothing is materialized: each part is a row-wise filter,
so both stay lazy.

```python
import batcher as bt

events = bt.from_pydict(
    {
        "event_id": list(range(1, 13)),
        "user_id": [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5],
        "ts": [1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 1, 2],
        "clicked": [1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0],
    }
)

train, test = events.ml.train_test_split(0.25, seed=42, key="event_id")
print(train.count() + test.count(), test.count())
# 12 2
```

Sizes are binomial around `test_size * n`, not exact. That is the price of a hash-keyed
assignment, and it is the right trade. An exact split needs a global shuffle or a counter,
neither of which survives being distributed.

## Leak 1: the same entity on both sides

A row-level split puts three of user 3's events in train and one in test. The model
memorizes user 3, scores itself on user 3, and reports a number nobody can reproduce on a
new user. Split by the *entity*, not the row: pass `key="user_id"` and every event of a
user hashes to the same part.

```python
train, test = events.ml.train_test_split(0.25, seed=2, key="user_id")
train_users = set(train.to_pydict()["user_id"])
test_users = set(test.to_pydict()["user_id"])
print(sorted(train_users), sorted(test_users), train_users & test_users)
# [1, 2, 3, 5] [4] set()
```

The intersection is empty by construction, because the assignment is a function of
`user_id` alone. The part sizes now follow the *users*, not the rows, so a whale with
10,000 events moves as one block, which is exactly what you want and also why the split is
lumpier. Check the row counts, not only the entity counts.

Group-splitting is the correct default for anything with a repeated actor: users, patient
IDs, source documents, product SKUs, or a `session_id` in clickstream data.

## Leak 2: the split moves when you touch a feature

The default (`key=None`) hashes **every column**. Recompute a feature, round a float
differently, add a column, and rows change parts. The two experiments you were comparing were
never evaluated on the same test set, and nothing tells you.

```python
from batcher import col

featured = events.with_columns(score=col("ts") * 2)
a, _ = featured.ml.train_test_split(0.25, seed=42)

rescored = events.with_columns(score=col("ts") * 3)  # same rows, new feature
b, _ = rescored.ml.train_test_split(0.25, seed=42)

print(sorted(a.to_pydict()["event_id"]) == sorted(b.to_pydict()["event_id"]))
# False
```

With `key="event_id"` the assignment ignores the feature columns entirely, so the split
survives any amount of feature engineering:

```python
a, _ = featured.ml.train_test_split(0.25, seed=42, key="event_id")
b, _ = rescored.ml.train_test_split(0.25, seed=42, key="event_id")
print(sorted(a.to_pydict()["event_id"]) == sorted(b.to_pydict()["event_id"]))
# True
```

Hashing one integer column is also cheaper than hashing twelve, and it does not depend on
how a float renders as text. Name the key on any real corpus.

## Leak 3: the future in the training set

For anything time-ordered (churn, fraud, demand, next-click) a random split trains on Tuesday
to predict Monday. Nothing about it is random in production, where the model only
ever sees the past. Cut on time instead, with an ordinary filter.

```python
cutoff = 3
past = events.filter(col("ts") < cutoff)
future = events.filter(col("ts") >= cutoff)
print(past.count(), future.count())
# 10 2
```

Hold out a *gap* between the two when the label itself takes time to materialize (a
30-day churn label needs 30 days to be known, so training rows within 30 days of the
cutoff have labels drawn from the test period). Two filters, one gap, no leak.

## Three-way splits

`random_split` generalizes to validation sets and takes the same `key` and `seed`.

```python
parts = events.ml.random_split([0.6, 0.2, 0.2], seed=3, key="event_id")
print([p.count() for p in parts])
# [7, 3, 2]
```

The parts are disjoint and together cover every row, so a `count()` over the three always
returns the input row count. On twelve rows the sizes wander a long way from 60/20/20;
on a million they will not.

## Reproducing a split later

:::{dropdown} There is nothing to checkpoint

The split is a pure function of `(key values, seed, test_size)`. There is no state to
checkpoint and no index list to store: the same call on the same rows, months later, on a
cluster of a different size, produces the same parts.

```python
again_train, again_test = events.ml.train_test_split(0.25, seed=2, key="user_id")
print(sorted(again_test.to_pydict()["event_id"]))
# [9, 10]
```

That property is why the split composes with the rest of the engine. Fit a
[preprocessor](feature-pipeline.md) on `train`, transform both parts, and the fitted
statistics stay attached to the same rows on every run.
:::

## Before you split: deduplicate

:::{important}
A split cannot separate two rows that are the same row. If the corpus contains the same
document twice (a repost, a re-crawl, the same article behind a different header) one copy
will land in train and the other in test, and your held-out score is a memorization score.
Deduplicate first, then split.
:::

```python
# docs: skip
clean = events.ml.drop_near_duplicates("text", threshold=0.8)
train, test = clean.ml.train_test_split(0.2, seed=42, key="doc_id")
```

[Training-data dedup](training-data-dedup.md) covers the fuzzy case, which is the one that
matters on a real corpus.

## See also

- [Feature pipeline](feature-pipeline.md): fit on train, transform both parts.
- [Training-data dedup](training-data-dedup.md): remove the twins before splitting.
- [Recommender features](recommender-features.md): the point-in-time filter that is the
  time-split's cousin.
- [Preprocessors](../../ml/preprocessors/index.md): what each `fit` learns, and from which split.
- [Streaming for training](../../ml/streaming.md): deterministic, resumable sample order
  across DDP ranks.
- [Data loaders](../../ml/data-loaders.md): handing `train` and `test` to a training loop.
- [ML API reference](../../api/ml.md): `train_test_split`, `random_split`,
  `drop_near_duplicates`.
- [Sampling](../../user-guide/sampling.md): the sampling surface the split is built on.
