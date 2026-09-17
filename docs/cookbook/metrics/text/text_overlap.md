# Reference overlap

These are the reference-based scores you can compute in the engine, with no embedding call and no GPU.

Use exact match for closed-form answers and normalized exact match when case and whitespace don't matter. Use token-set overlap for short free text, and character n-gram overlap when wording varies but content shouldn't. In the script, token-set F1 gives credit to an answer that says the right thing in a different order, where exact match gives none. `length_ratio` tells you whether answers run systematically long or short.

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
