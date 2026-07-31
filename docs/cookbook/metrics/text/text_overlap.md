# Reference overlap

These are the reference-based scores you can compute in the engine: exact match for closed-form answers, token-set overlap for short free text, and character n-gram overlap when wording varies but content should not. No embedding call, no GPU.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/metrics/text_overlap.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/metrics/text_overlap.py
```

## See also

- {doc}`/cookbook/metrics/text/text_length`: length and readability distribution over a text column.
- {doc}`/cookbook/metrics/text/text_pii_safety`: PII leak rates over a text column.
- {doc}`/ml/evaluation/evaluation`: scoring a model, per segment, in one pass.
- {doc}`/api/models/metrics`: the complete metric vocabulary.
