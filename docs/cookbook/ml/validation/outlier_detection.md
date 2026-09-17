# Outlier detection

A per-column rule misses a row that is unremarkable on every axis and absurd as a combination. Mahalanobis distance catches it, which is why it is the one to reach for on correlated features.

The script applies the IQR and z-score rules per column, then ranks rows by `mahalanobis_distance`. The row it puts first is 155 cm tall and weighs 200 kg, and its height alone looks perfectly ordinary.

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
- {doc}`/ml/evaluation/statistics-and-drift`: the outlier rules and the statistics around them, in full.
