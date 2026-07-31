# Text features

Before reaching for an embedding, check whether cheap features answer the question. Length, character mix, and token counts separate a lot of classes on their own, and they cost a scan rather than a GPU.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/ml/text_features.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/ml/text_features.py
```

## See also

- {doc}`/cookbook/ml/preprocessing/preprocessing_scaling`: scaling numeric features, and why the choice of scaler matters.
- {doc}`/cookbook/ml/inference/vector_search`: vector search over an embedding column, in the engine.
- {doc}`/ml/index`: the ML surface these recipes sit on.
- {doc}`/ml/preparing/preprocessors/index`: the fit and transform steps most pipelines start with.
