# Preprocessor chains

A ``Chain`` fits its steps in order and applies them in order, so the whole feature pipeline is a single object you fit on train and apply to everything else. That is what stops a validation set being scaled by its own statistics.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/ml/preprocessing_chain.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/ml/preprocessing_chain.py
```

## See also

- {doc}`/cookbook/ml/preprocessing/preprocessing_binning`: discretizing, clipping, and reshaping the distribution of a numeric column.
- {doc}`/cookbook/ml/preprocessing/preprocessing_encoding`: turning categories into numbers, and picking the encoder by cardinality.
- {doc}`/ml/index`: the ML surface these recipes sit on.
- {doc}`/ml/preparing/preprocessors/index`: the fit and transform steps most pipelines start with.
