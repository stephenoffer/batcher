# Model selection

`cross_val_score` takes a `fit` and a `predict` callable, so it works with the built-in estimators or with anything you wrap. Pass `key=` when rows share a group that must not straddle a fold. That is the difference between an honest score and a leak.

The script scores a ridge model with plain and group-aware folds, then produces out-of-fold predictions, a learning curve, permutation importance, and partial dependence.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/ml/model_selection.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/ml/model_selection.py
```

## See also

- {doc}`/cookbook/ml/estimators/linear_models`: Ridge, Lasso, and ElasticNet.
- {doc}`/cookbook/ml/validation/outlier_detection`: per-column rules and a multivariate distance.
- {doc}`/ml/index`: the ML surface these recipes sit on.
- {doc}`/ml/evaluation/model-selection`: cross-validation and hyperparameter search, in full.
