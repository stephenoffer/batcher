# Outlier detection

A per-column rule misses the point that a row can be unremarkable on every axis and still be absurd as a combination. Mahalanobis distance catches that, which is why it is the one to reach for on correlated features.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/ml/outlier_detection.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/ml/outlier_detection.py
```

## See also

- {doc}`/cookbook/ml/validation/model_selection`: cross-validation, learning curves, and feature importance, all in the engine.
- {doc}`/cookbook/ml/preprocessing/preprocessing_binning`: discretizing, clipping, and reshaping the distribution of a numeric column.
- {doc}`/ml/index`: the ML surface these recipes sit on.
- {doc}`/ml/preparing/preprocessors/index`: the fit and transform steps most pipelines start with.
