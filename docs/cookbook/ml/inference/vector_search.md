# Vector search

Keeping vectors as a list column means retrieval is a projection plus a top-N, composable with any other filter. That is what lets you pre-filter by metadata *before* scoring, which is both faster and more correct than scoring everything and filtering after.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/ml/vector_search.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/ml/vector_search.py
```

## See also

- {doc}`/cookbook/ml/preprocessing/text_features`: turning raw text into model-ready features without a model.
- {doc}`/cookbook/ml/preprocessing/preprocessing_scaling`: scaling numeric features, and why the choice of scaler matters.
- {doc}`/ml/index`: the ML surface these recipes sit on.
- {doc}`/ml/preparing/preprocessors/index`: the fit and transform steps most pipelines start with.
