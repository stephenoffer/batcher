# Recommender features

A recommender's training row is a `(user, item, label)` triple with features attached. The
features are the whole job, and the failure mode is always the same: a feature computed
over the entire event log, including events that happened *after* the label. The model
learns that users who click a lot are users who clicked, offline AUC goes to 0.95, and the
online lift is zero.

Point-in-time correctness is not a library feature. It is a filter you have to write, and
this page shows where it goes.

## The event log

```python
import batcher as bt

events = bt.from_pydict(
    {
        "user_id": [1, 1, 1, 2, 2, 2, 3, 3],
        "item_id": [10, 11, 12, 10, 13, 11, 12, 10],
        "day": [1, 2, 5, 1, 3, 6, 2, 7],
        "clicked": [1, 0, 1, 1, 1, 0, 0, 1],
    }
)

CUTOFF = 5  # features from day < 5; labels from day >= 5
history = events.filter(bt.col("day") < CUTOFF)
labels = events.filter(bt.col("day") >= CUTOFF)
print(history.count(), labels.count())
# 5 3
```

:::{warning}
Every feature below is computed from `history` only. That single filter is what separates a
model you can deploy from one you cannot. Compute one of them over the whole event log and the
feature carries the answer inside it: offline AUC climbs, online lift is zero, and no error is
raised anywhere in between.
:::

The features come in four shapes, and each has its own way of leaking:

| Feature | Operator | What goes wrong |
| --- | --- | --- |
| Per-user counts and rates | `group_by().agg(...)` | computed over the label window too, so the count knows the future |
| "Time since last event", "previous item" | `ds.window(...)` with `lag` | the first event's null filled with 0, which claims the user acted today |
| Per-item history | `group_by("item_id")` + a `left` join | an inner join, which deletes the cold-start rows you most need |
| Item tags | `MultiHotEncoder` | refit at serving time, which shifts every column index |

## Aggregate features per user

`group_by().agg(...)` is a mergeable aggregation: it runs partial → combine → finalize, so
the same code gives the same answer on one core, on 96, or across a cluster, and it spills
instead of dying when the group set does not fit in memory.

```python
from batcher import col

user_features = history.group_by("user_id").agg(
    impressions=col("item_id").count(),
    clicks=col("clicked").sum(),
    distinct_items=col("item_id").n_unique(),
)
print(user_features.sort("user_id").to_pydict())
# {'user_id': [1, 2, 3], 'impressions': [2, 2, 1], 'clicks': [1, 2, 0], 'distinct_items': [2, 2, 1]}
```

A click-through rate is a ratio of two of those, and a ratio over small counts is noise: a
user with one impression and one click has a CTR of 1.0, which will dominate any model that
believes it. Smooth it toward the global rate with a pseudo-count.

```python
ctr = user_features.with_columns(
    user_ctr=(col("clicks") + 1.0) / (col("impressions") + 10.0)  # Laplace-smoothed
)
print([round(v, 4) for v in ctr.sort("user_id").to_pydict()["user_ctr"]])
# [0.1667, 0.25, 0.0909]
```

## Sequence features come from windows

"How long since this user's last event" and "what did they see before this one" are window
functions over the partition, not aggregates. `ds.window` computes them in one operator, in
the engine.

```python
sequenced = history.window(
    partition_by=["user_id"],
    order_by=[("day", False)],
    functions={"prev_day": ("lag", "day", 1), "prev_item": ("lag", "item_id", 1)},
).with_columns(days_since_last=col("day") - col("prev_day"))

out = sequenced.sort("user_id", "day").to_pydict()
print(out["user_id"], out["day"], out["days_since_last"])
# [1, 1, 2, 2, 3] [1, 2, 1, 3, 2] [None, 1, None, 2, None]
```

:::{note}
The null on a user's first event is correct and should stay a null. Filling it with 0 tells
the model "this user acted zero days ago", which is the opposite of the truth. Impute it
with a sentinel, or let the model handle the null.
:::

## Item-side features and the join

Item features are the same shape, computed over the same history window, and joined onto
the label rows.

```python
item_features = history.group_by("item_id").agg(
    item_impressions=col("item_id").count(),
    item_clicks=col("clicked").sum(),
)

training = (
    labels.select("user_id", "item_id", "day", "clicked")
    .join(user_features, on="user_id", how="left")
    .join(item_features, on="item_id", how="left")
)
print(training.sort("user_id", "item_id").to_pydict()["item_impressions"])
# [1, 1, 2]
```

:::{tip}
A `left` join, not an inner one. An item that appears for the first time in the label
window has no history, and an inner join would silently delete exactly the cold-start rows
you most need to handle. Keep them and let the nulls be a feature.
:::

## Item tags: multi-hot, learned on history

:::{dropdown} One indicator column per tag, and why you never refit at serving time

An item's tag list is a `list<string>` column. `MultiHotEncoder` learns the distinct tags
over the training data and emits one indicator column per tag.

```python
from batcher.ml import MultiHotEncoder

items = bt.from_pydict(
    {
        "item_id": [10, 11, 12, 13],
        "tags": [["sports", "news"], ["news"], ["tech"], ["tech", "news"]],
    }
)
encoded = MultiHotEncoder("tags").fit(items).transform(items)
print(encoded.collect().column_names)
# ['item_id', 'tags', 'tags_news', 'tags_sports', 'tags_tech']
```

Fit on the items present in the training window. A tag that first appears in production
gets an all-zero row rather than a new column that shifts every other index, which is what
would happen if you refit at serving time.
:::

## Negative sampling

An impression log is mostly negatives already. An *interaction* log (purchases, likes) is
all positives, and a model trained on it learns to predict 1. Sample negatives from the
items the user did not touch: a cross join to the candidate space, an anti-join against the
positives, and a `sample` to size it.

```python
positives = events.filter(col("clicked") == 1).select("user_id", "item_id")
users = events.select("user_id").distinct()
catalog = events.select("item_id").distinct()

negatives = (
    users.cross_join(catalog)
    .join(positives, on=["user_id", "item_id"], how="anti")
    .sample(0.5, seed=7)
    .with_columns(clicked=bt.lit(0))
)
print(positives.count(), negatives.count())
# 5 4
```

The cross join is the part that scales badly on a real catalogue. A million users times a
million items is not a table you want. In practice you sample the candidates first (from a
popularity distribution, or from a retrieval model's top-k) and cross-join against *that*.
The operators are the same; only the size of `catalog` changes.

## Then split, then fit

The features are built. Split by user so no user appears on both sides, fit the scalers on
train only, and stream the result into the model.

```python
labelled = positives.with_columns(clicked=bt.lit(1)).union(negatives)
train, test = labelled.ml.train_test_split(0.25, seed=0, key="user_id")
print(train.count(), test.count())
# 6 3
```

Splitting on `user_id` rather than on the row is the difference between measuring
generalization to a new user and measuring memorization of an old one. See
[train/test split](train-test-split.md) for the other two ways that goes wrong.

## See also

- [Feature pipeline](feature-pipeline.md): scale, encode, and assemble these columns.
- [Train/test split](train-test-split.md): entity splits, time splits, and leakage.
- [Window functions](../../user-guide/window-functions.md): frames, ranks, and lag/lead in full.
- [Aggregations](../../user-guide/aggregations.md): the mergeable aggregate surface.
- [Joins](../../user-guide/joins.md): left, anti, and cross joins, and what each one costs.
- [Preprocessors](../../ml/preprocessors.md): `MultiHotEncoder` and the rest of the estimators.
- [Data loaders](../../ml/data-loaders.md): getting the finished rows into a training loop.
- [ML API reference](../../api/ml.md): `ds.ml.train_test_split` and the `batcher.ml` estimators.
- [Mergeable algebra](../../deep-dives/mergeable-algebra.md): why these aggregates give the
  same answer on one core and on a cluster.
- [Sessionization](../analytics/sessionization.md): the window-function recipe this one borrows
  its sequence features from.
