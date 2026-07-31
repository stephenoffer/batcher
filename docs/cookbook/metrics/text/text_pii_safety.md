# PII and safety rates

Run this over model output *and* over training data. On output it tells you whether the model is emitting personal data; on input it tells you whether you are about to train on it. Both are one aggregate pass, so it is cheap enough to run on every batch.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/metrics/text_pii_safety.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/metrics/text_pii_safety.py
```

## See also

- {doc}`/cookbook/metrics/text/text_overlap`: comparing a generated answer against a reference, without a model.
- {doc}`/cookbook/metrics/text/text_quality`: what fraction of a text column looks broken.
- {doc}`/ml/evaluation/evaluation`: scoring a model, per segment, in one pass.
- {doc}`/api/models/metrics`: the complete metric vocabulary.
