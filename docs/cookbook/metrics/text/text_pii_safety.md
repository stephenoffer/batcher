# PII and safety rates

Run these rates over model output *and* over training data. On output they tell you whether the model emits personal data. On input they tell you whether you are about to train on it. Either way it is one aggregate pass, cheap enough to run on every batch.

The script counts emails, phone numbers, and card-like and SSN-like strings, plus a denylist of your own terms with `contains_any_rate`. `pii_rate` combines email and phone only, so cards and SSNs are counted by their own metrics.

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
