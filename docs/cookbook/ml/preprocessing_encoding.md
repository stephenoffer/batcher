# Turning categories into numbers, and picking the encoder by cardinality

One-hot is fine for a handful of categories and catastrophic for a million user ids. The alternatives trade information for width: ordinal keeps one column, frequency and target encoding keep one column carrying signal, and hashing bounds the width outright.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/ml/preprocessing_encoding.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/ml/preprocessing_encoding.py
```

## See also

- {doc}`preprocessing_chain`: chaining preprocessors into one fitted pipeline.
- {doc}`preprocessing_imputation`: filling missing values, and keeping the fact that they were missing.
- {doc}`../../ml/index`: the ML surface these recipes sit on.
- {doc}`../../ml/preprocessors/index`: the fit and transform steps most pipelines start with.
