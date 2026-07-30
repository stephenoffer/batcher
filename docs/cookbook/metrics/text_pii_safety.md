# PII leak rates over a text column

Run this over model output *and* over training data. On output it tells you whether the model is emitting personal data; on input it tells you whether you are about to train on it. Both are one aggregate pass, so it is cheap enough to run on every batch.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/metrics/text_pii_safety.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/metrics/text_pii_safety.py
```

## See also

- {doc}`text_overlap`: comparing a generated answer against a reference, without a model.
- {doc}`text_quality`: corpus hygiene rates: what fraction of a text column looks broken.
- {doc}`../../ml/evaluation`: scoring a model, per segment, in one pass.
- {doc}`../../api/metrics`: the complete metric vocabulary.
