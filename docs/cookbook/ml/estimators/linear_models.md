# Regularized linear models

`Ridge`, `Lasso`, and `ElasticNet` follow the two-step shape every Batcher estimator uses. `fit(ds)` returns a fitted model, and `predict(ds)` returns a new Dataset with a `prediction` column appended. Fitting reads the data through the engine, so the training set never has to fit in memory as a NumPy array.

The script recovers known coefficients, renames the output with `output_column=` to put two models side by side, and scores the fit with the `r2` and `rmse` aggregates.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/ml/linear_models.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/ml/linear_models.py
```

## See also

- {doc}`/cookbook/ml/validation/imbalance_and_sampling`: measure it, then resample or reweight.
- {doc}`/cookbook/ml/validation/model_selection`: cross-validation, learning curves, and feature importance, all in the engine.
- {doc}`/ml/index`: the ML surface these recipes sit on.
- {doc}`/ml/evaluation/index`: scoring the model once it is fitted, as engine expressions.
