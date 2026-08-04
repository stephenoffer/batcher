# Ensembling

This page describes how to combine several models into one prediction on Batcher, and how
to do it without the leak that makes a stacked ensemble look better than it is.

Ensembling works because models make *different* mistakes. Two models that are individually
mediocre but wrong about different rows combine into something better than either; two
models that are wrong about the same rows combine into the same mistakes with more compute.
That is the thing to check before reaching for any of this.

## Averaging, first

{py:func}`blend_predictions <batcher.ml.blend_predictions>` takes prediction columns already
in the frame and appends their weighted mean. There is no fit, no held-out split, and no
meta-model, so there is nothing to leak and nothing to tune:

```python
import batcher as bt
from batcher.ml.ensemble import blend_predictions

scored = bt.from_pydict({"model_a": [0.2, 0.9], "model_b": [0.4, 0.7]})
print(blend_predictions(scored, ["model_a", "model_b"]).to_pydict()["prediction"])
# [0.30000000000000004, 0.8]
```

Weights are normalized to sum to one, so `[3, 1]` and `[0.75, 0.25]` mean the same thing and
the blend stays on the scale of its inputs:

```python
weighted = blend_predictions(scored, ["model_a", "model_b"], weights=[3, 1])
print(weighted.to_pydict()["prediction"])
# [0.25, 0.8500000000000001]
```

Blend *probabilities*, not hard class labels. The mean of labels 0 and 2 is 1, which may be
a class neither model predicted. Blend the probability columns and threshold afterwards.

## Stacking, when a fixed average isn't enough

A blend applies the same weights everywhere. A meta-model can learn that one base model is
the one to trust on short documents and another on long ones, which no fixed average can
express. {py:class}`StackingEnsemble <batcher.ml.StackingEnsemble>` fits that meta-model.

Base models are `(fit, predict)` callable pairs — the same shape
{py:func}`cross_val_score <batcher.ml.cross_val_score>` takes — so a Batcher estimator, a
scikit-learn one, or a whole preprocessing pipeline all compose without an adapter:

```python
from batcher.ml import LinearRegression, Ridge
from batcher.ml.ensemble import StackingEnsemble

ds = bt.from_pydict({"x": [float(i) for i in range(40)], "y": [2.0 * i for i in range(40)]})
bases = {
    "ridge": (lambda d: Ridge(["x"], "y", alpha=1.0).fit(d), lambda m, d: m.predict(d)),
    "ols": (lambda d: LinearRegression(["x"], "y").fit(d), lambda m, d: m.predict(d)),
}
meta = (
    lambda d: LinearRegression(["ridge", "ols"], "y").fit(d),
    lambda m, d: m.predict(d),
)
stack = StackingEnsemble(bases, meta, k=4, key="x").fit(ds)
print(stack.predict(ds).count())
# 40
```

Each base model's name becomes the feature column the meta-model sees, so `meta` is fitted
against `ridge` and `ols` here rather than against `x`.

## Why the out-of-fold part matters

This is the one thing to get right. If the meta-model trains on predictions the base models
made about rows they were *fitted on*, it learns to trust whichever base model memorized
hardest — and that is the model that will do worst on data it has not seen. The ensemble
then scores beautifully in development and badly in production, with nothing failing in
between.

`StackingEnsemble` avoids that by fitting the meta-model on out-of-fold predictions: every
row is scored by base models that never saw it. You can build that table directly with
{py:func}`out_of_fold_features <batcher.ml.out_of_fold_features>`, which is useful when you
want to inspect the base models' agreement before committing to a meta-model:

```python
from batcher.ml.ensemble import out_of_fold_features

features = out_of_fold_features(ds, bases, k=4, key="x")
print(sorted(features.columns))
# ['ols', 'ridge', 'x', 'y']
```

All the base models are fitted and scored inside a *single* fold loop, so each row's columns
describe that row without a join and without needing a row key to join on. Calling
{py:func}`cross_val_predict <batcher.ml.cross_val_predict>` once per model would give
datasets whose row orders do not correspond.

Pass `key=` naming the columns that identify a row, so the fold assignment is stable when a
feature is re-derived, and `stratify=` for an imbalanced target.

## Two sets of fitted models

`fit` produces two different things, and the difference is deliberate:

- The **out-of-fold columns** come from `k` fold-fitted copies of each base model. They exist
  only to give the meta-model an honest training table, and each of them saw a fraction of
  the data.
- The **prediction path** uses one copy of each base model refitted on the whole training
  split. Scoring a new row with a fold-fitted copy would throw away most of the training
  data for no reason.

So a `k=5` stack over three base models costs eighteen fits: fifteen for the features, three
for the refit.

## Requirements and limitations

- Every base model's `predict` must append the same column name, `prediction` by default.
  A base model that writes nothing is reported by name rather than producing a silently
  empty feature.
- A base model may not be named `prediction`, because its feature column would collide with
  the column the models write into.
- Row order is not preserved by `out_of_fold_features`: it is the union of the scored folds.
  Sort or join on your key if order matters.
- Stacking multiplies the fit cost. Blend first, and only stack if the blend leaves
  something on the table.

## See also

- {doc}`/ml/evaluation/evaluation` for the metrics to compare a blend against its bases.
- {doc}`/api/models/ml-models` for the full reference.
