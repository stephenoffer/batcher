# Splits and resampling

This page covers the preparation an honest evaluation needs: reshaping a class balance a model would otherwise ignore, holding out a test set that still contains the rare class, and building cross-validation folds that leak nothing. The statistics and drift measures that surround a model are on {doc}`statistics-and-drift`.

## Resampling for imbalanced learning

A classifier trained on a 1%-positive dataset learns to predict "negative" and scores well on
accuracy while being useless. `batcher.ml.sampling` reshapes the class balance as a relational
operation: an exact content-hashed filter or concatenation, never a driver-side shuffle. That is
what lets it run over a dataset larger than memory.

```python
import batcher as bt
from batcher.ml.sampling import class_counts, class_weights, oversample, undersample

ds = bt.from_pydict({"y": [0] * 100 + [1] * 10, "x": list(range(110))})
print(class_counts(undersample(ds, "y"), "y"))  # exactly balanced by discarding
print(class_counts(oversample(ds, "y"), "y"))  # exactly balanced by duplicating
```

`undersample` discards majority rows. `oversample` duplicates minority rows
deterministically, and `balanced_sample` moves every class to the median count. When the model supports it, prefer
{py:meth}`class_weights <batcher.Dataset.class_weights>` (a `{class: weight}` dict for the model's ``class_weight``) or `sample_weights`
(a per-row weight column). Both rebalance the *loss* without discarding or duplicating a
single row. `class_counts` is the first thing to look at.

{py:func}`smote <batcher.ml.smote>` is the alternative to duplicating. `oversample` repeats
minority rows, so a model can still memorize the exact points and tighten its boundary around
them rather than around the region they occupy. SMOTE makes *new* points instead, each on the
segment between a real minority row and one of its nearest minority neighbours:

```python
from batcher.ml import smote

rare = bt.from_pydict(
    {"x": [0.0, 0.1, 0.2, 5.0, 5.1, 5.2, 5.3, 5.4], "label": ["rare"] * 3 + ["common"] * 5}
)
print(smote(rare, "label", minority="rare", features=["x"]).count())
# 10
```

The synthetic rows are interpolations, never extrapolations, so they stay inside the
minority region. Both random draws are content hashes of the row, so the same input produces
the same synthetic rows however the data is partitioned. An imbalanced experiment stays
repeatable.

Two things to know before using it. Scale the features first, because distance decides which
neighbours a row is drawn towards and therefore where the new points land. And only the
`features` and the label are filled: any other column is null on a synthetic row, because
there is no honest value to interpolate for an identifier or a free-text field.

`stratified_sample` is the different tool for a different job: it keeps the same fraction of *every* stratum rather than equalizing them, so it shrinks a dataset for a quick experiment while preserving its class balance. You get a proportional 10% sample rather than 10% of the whole, which would starve the rare classes.

```python
from batcher.ml.sampling import class_counts, stratified_sample

ds = bt.from_pydict({"y": [0] * 100 + [1] * 20, "x": list(range(120))})
print(class_counts(stratified_sample(ds, "y", 0.5, seed=1), "y"))  # {0: 50, 1: 10}
```

## Holding out a test set that still has the rare class in it

`ds.ml.train_test_split` assigns each row by a content hash, which makes the split proportional in expectation and nothing more. On 200 rows with ten positives and a quarter held out, the test half should get two or three. Across seeds it gets between one and four. One positive makes precision, recall and AUC meaningless, and nothing reports a problem.

`stratify=` names a column whose distribution to hold constant across both halves:

```python
sales = bt.from_pydict(
    {
        "amount": [float(i) for i in range(200)],
        "fraud": [1 if i % 20 == 0 else 0 for i in range(200)],
    }
)

plain = sales.ml.train_test_split(test_size=0.25, seed=3)[1]
kept = sales.ml.train_test_split(test_size=0.25, seed=3, stratify="fraud")[1]
print(sum(plain.to_pydict()["fraud"]), sum(kept.to_pydict()["fraud"]))
# 4 3
```

The stratified count is the same for every seed, because it is a property of the split rather than of the draw. Every label with at least two rows reaches both halves, and the cut rounds towards putting a rare class in the test half rather than away from it. A label with a single row goes to train, since a model that never saw the class is the worse of the two mistakes.

Reach for {py:func}`stratified_split <batcher.ml.splitting.stratified_split>` directly when you want the same behaviour outside the `ds.ml` surface.

## Cross-validation splits

A fold here is a **content hash** of each row compared against fold boundaries, never a materialized shuffle. That means a fold is an ordinary row-wise filter, the assignment is identical however the data is partitioned, and the training half of a fold stays lazy until something reads it.

```python
ds = bt.range(0, 1000)
folds = ds.ml.kfold(5, key="value")
print(sum(validate.count() for _, validate in folds))
```

Two options select the variant your data needs, and choosing correctly is usually the difference between a trustworthy score and a misleading one:

`stratify=` keeps each label's proportion identical in every fold. Use it whenever the label is imbalanced, or the fold-to-fold variance in the score measures the split rather than the model.

```python
ds = bt.from_pydict({"y": [0] * 90 + [1] * 10, "x": list(range(100))})
folds = ds.ml.kfold(5, key="x", stratify="y")
print([v.filter(bt.col("y") == 1).count() for _, v in folds])
```

`group=` keeps every row of a group in the same fold. Use it whenever rows repeat an entity such as a user, a patient, a session, or a document. Without it the model memorizes the entity rather than the pattern, cross-validation looks excellent, and production does not. This is the most common silent leak in applied ML.

For a time series, neither applies: a random fold puts next week's rows in the training set, so the model sees the future and the validation score is one no deployment will reproduce.

```python
ds = bt.from_pydict({"t": list(range(100)), "x": list(range(100))})
print([(train.count(), validate.count()) for train, validate in ds.ml.time_series_split("t", 4)])
```

`expanding=True`, the default, grows the training window with each split, matching a model retrained on all history. `expanding=False` slides a fixed-width window, matching one that deliberately forgets.

`batcher.ml.model_selection` runs the loop end to end. `cross_val_score` fits and scores a
model on each fold, and the spread across those folds is the honesty a single number hides.
`cross_val_predict` gives every row its out-of-fold prediction, which is the unbiased input
a stacking ensemble needs. `learning_curve` scores against training-set size, to answer
whether more data would help. Each takes a `fit` and a `predict` callable, so any
scikit-learn-style model composes.

{py:func}`batcher.ml.splitting.fold_column <batcher.ml.splitting.fold_column>` is the primitive underneath. Reach for it when the split should outlive the pipeline that created it: it writes one column that every downstream job can filter on without re-deriving the assignment.

## Requirements and limitations

Fold sizes are binomial around `n / k` rather than exact, as with any hash-keyed split, and `group_kfold`'s folds vary further because groups differ in size.

## See also

- {doc}`model-selection`: the search loop that runs over the folds built here.
- {doc}`statistics-and-drift`: the statistics, outlier rules, and drift measures around a model.
- {doc}`/ml/evaluation/evaluation`: score a model once you have a trustworthy split.
- {doc}`/cookbook/ml/validation/imbalance_and_sampling`: the resampling functions on this page, as a runnable script.
