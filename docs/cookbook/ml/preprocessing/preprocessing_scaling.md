# Feature scaling

Every scaler splits `fit` from `transform`. The statistics come from the training set and are then *applied* to validation and production data. Fitting on everything is the classic leak, and the API makes the correct order the easy one.

The script compares `StandardScaler` and `MinMaxScaler` with `RobustScaler`, which a single outlier of 100.0 barely moves, and `MaxAbsScaler`. Then `Normalizer` scales each row, rather than each column, to unit length.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/ml/preprocessing_scaling.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/ml/preprocessing_scaling.py
```

## See also

- {doc}`/cookbook/ml/preprocessing/preprocessing_imputation`: filling missing values, and keeping the fact that they were missing.
- {doc}`/cookbook/ml/preprocessing/text_features`: turning raw text into model-ready features without a model.
- {doc}`/ml/index`: the ML surface these recipes sit on.
- {doc}`/ml/preparing/preprocessors/index`: the fit and transform steps most pipelines start with.
