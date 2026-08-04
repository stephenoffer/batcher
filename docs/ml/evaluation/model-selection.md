# Model selection and hyperparameter search

This page describes how to cross-validate a model on Batcher, and how to search a
hyperparameter space without hand-writing the loop.

Everything here takes `fit` and `predict` as callables rather than requiring a particular
model class, so a Batcher estimator, a scikit-learn one, or a closure that fits a whole
preprocessing pipeline all compose the same way.

## Cross-validated scoring

{py:func}`cross_val_score <batcher.ml.cross_val_score>` fits on each fold's training part,
predicts the held-out part, and returns one score per fold:

```python
import batcher as bt
from batcher.ml import Ridge
from batcher.ml.metrics import evaluate
from batcher.ml.model_selection import cross_val_score

ds = bt.from_pydict({"x": [float(i) for i in range(40)], "y": [2.0 * i for i in range(40)]})

def r2(scored, y_true, y_pred):
    return evaluate(scored, y_true, y_pred=y_pred, task="regression", metrics=["r2"])["r2"]

scores = cross_val_score(
    ds,
    lambda train: Ridge(["x"], "y", alpha=0.1).fit(train),
    lambda model, part: model.predict(part),
    y_true="y",
    metric=r2,
    k=4,
    key="x",
)
print(len(scores), all(s > 0.99 for s in scores))
# 4 True
```

Read the spread, not just the mean. A model with a great average and a wide spread across
folds is one split away from looking bad, and the mean alone hides that.

Pass `key=` naming the columns that identify a row. The folds are content-hash filters, so
hashing only the identity columns keeps the assignment stable when you re-derive a feature.
Pass `stratify=` for an imbalanced target, so each fold carries the same class balance.

## Searching a parameter space

A search is the same fold loop run once per parameter combination.
{py:func}`param_grid <batcher.ml.param_grid>` builds the combinations and
{py:func}`grid_search <batcher.ml.grid_search>` scores them:

```python
from batcher.ml.model_selection import grid_search, param_grid

found = grid_search(
    ds,
    lambda train, params: Ridge(["x"], "y", alpha=params["alpha"]).fit(train),
    lambda model, part: model.predict(part),
    y_true="y",
    metric=r2,
    grid=param_grid(alpha=[0.01, 1.0, 100.0]),
    k=4,
    key="x",
)
print(found.best_params)
# {'alpha': 0.01}
```

Every combination is scored on the *same* folds. That makes the comparison paired: the same
rows train and validate each candidate, so a difference between two scores is a difference
between the candidates rather than fold-assignment luck.

Nothing is refitted on the full dataset afterwards. What comes back is the winning
parameters, which keeps the search independent of whatever `fit` builds — refit yourself
with one more call.

## Reading the whole search, not just the winner

A best score reported alone hides whether the search found a peak or a plateau. A plateau
means the parameter did not matter, which is more useful than knowing which arbitrary point
on it won. `trials` carries every combination, best first:

```python
for trial in found.trials:
    print(trial["params"], round(trial["mean"], 4))
# {'alpha': 0.01} 1.0
# {'alpha': 1.0} 1.0
# {'alpha': 100.0} 0.9993
```

{py:meth}`SearchResult.to_dataset <batcher.ml.SearchResult.to_dataset>` gives the same thing
as a `Dataset`, one column per parameter plus `mean_score` and `std_score`, which is the
convenient shape for writing the search log next to the model:

```python
print(found.to_dataset().columns)
# ['alpha', 'mean_score', 'std_score']
```

## Minimizing a loss

`greater_is_better` decides the direction, and it defaults to maximizing. Hand a search an
error metric and leave the default alone, and it returns the *worst* combination —
confidently, with no error. Set it whenever the metric is a loss:

```python
def rmse(scored, y_true, y_pred):
    return evaluate(scored, y_true, y_pred=y_pred, task="regression", metrics=["rmse"])["rmse"]

by_error = grid_search(
    ds,
    lambda train, params: Ridge(["x"], "y", alpha=params["alpha"]).fit(train),
    lambda model, part: model.predict(part),
    y_true="y",
    metric=rmse,
    grid=param_grid(alpha=[0.01, 100.0]),
    greater_is_better=False,
    k=4,
    key="x",
)
print(by_error.best_params)
# {'alpha': 0.01}
```

## Random search, for more than a couple of parameters

Once three or more parameters are in play, a grid spends most of its budget re-testing the
ones that do not matter. {py:func}`random_search <batcher.ml.random_search>` gives every
parameter `n_iter` distinct values for the same number of fits.

A parameter's candidates are either a sequence, drawn from uniformly, or a callable taking
a `random.Random`, which is how a continuous range is expressed.
{py:func}`param_samples <batcher.ml.param_samples>` is the draw on its own, useful when you
want to inspect the combinations before spending fits on them:

```python
from batcher.ml.model_selection import param_samples

print(param_samples(2, seed=0, alpha=[0.01, 1.0], depth=lambda rng: rng.randint(1, 3)))
# [{'alpha': 1.0, 'depth': 2}, {'alpha': 0.01, 'depth': 2}]
```

Passing the same candidates to `random_search` runs the whole loop:

```python
from batcher.ml.model_selection import random_search

sampled = random_search(
    ds,
    lambda train, params: Ridge(["x"], "y", alpha=params["alpha"]).fit(train),
    lambda model, part: model.predict(part),
    y_true="y",
    metric=r2,
    distributions={"alpha": lambda rng: 10 ** rng.uniform(-3, 2)},
    n_iter=6,
    seed=0,
    k=4,
    key="x",
)
print(len(sampled.trials), sampled.best_score > 0.99)
# 6 True
```

`seed` fixes both the draw and the fold assignment, so a search is reproducible.

## Sizing the training set instead

Sometimes the question is not which parameters but whether more data would help.
{py:func}`learning_curve <batcher.ml.learning_curve>` scores at increasing training-set
sizes: a gap that is still closing means collect more, and a gap that has flattened means
the ceiling is the model.

{py:func}`validation_curve <batcher.ml.validation_curve>` is the one-parameter version of a
grid search, returning the score as a function of that parameter, which is the right shape
for plotting the peak rather than just naming it.

## Requirements and limitations

- A search costs one model fit per combination per fold. `grid_search` on a 3x3x3 grid with
  5 folds is 135 fits; size the grid against what a fit costs.
- Selecting hyperparameters on the same folds you report the score from is mildly
  optimistic, because the winner was chosen partly by fold noise. Hold out a final test
  split that the search never touches, or nest a second cross-validation around it.
- `SearchResult` is frozen, so a search result cannot be edited after the fact.
- `to_dataset` puts every parameter in a column, so a parameter whose values are not scalars
  (a list, a nested estimator) has no sensible column type. Read `trials` directly for
  those.

## See also

- {doc}`evaluation` for the metrics these searches optimize.
- {doc}`/ml/preparing/preprocessors/feature-selection` for pruning features rather than
  tuning parameters.
- {doc}`/api/models/ml-statistics` for the full reference.
