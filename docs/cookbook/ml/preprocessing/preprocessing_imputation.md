# Imputation

Imputing silently destroys information: "no value recorded" often predicts the target better than whatever you filled in. ``MissingIndicator`` keeps that signal as its own column, so impute *and* flag rather than choosing.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/ml/preprocessing_imputation.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/ml/preprocessing_imputation.py
```

## See also

- {doc}`/cookbook/ml/preprocessing/preprocessing_encoding`: turning categories into numbers, and picking the encoder by cardinality.
- {doc}`/cookbook/ml/preprocessing/preprocessing_scaling`: scaling numeric features, and why the choice of scaler matters.
- {doc}`/ml/index`: the ML surface these recipes sit on.
- {doc}`/ml/preparing/preprocessors/index`: the fit and transform steps most pipelines start with.
