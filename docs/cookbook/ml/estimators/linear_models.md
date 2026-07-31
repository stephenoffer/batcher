# Regularized linear models

Every estimator follows the same two-step shape: ``fit(ds)`` returns a fitted model and ``predict(ds)`` returns a new Dataset with the prediction column appended. Fitting reads the data through the engine, so the training set never has to fit in memory as a NumPy array.

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
- {doc}`/ml/preparing/preprocessors/index`: the fit and transform steps most pipelines start with.
