# Turning raw text into model-ready features without a model

Before reaching for an embedding, check whether cheap features answer the question. Length, character mix, and token counts separate a lot of classes on their own, and they cost a scan rather than a GPU.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/ml/text_features.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/ml/text_features.py
```
