# Corpus hygiene rates

One `select` over these aggregates gives you a scorecard for a whole generation run. Most return a rate in [0, 1]: blank outputs, all caps, stray or doubled whitespace, non-ASCII text, URLs, and outputs that are too short or too long.

They are the numbers to watch between model versions. A jump in `empty_or_whitespace_rate` is a broken prompt, not a worse model.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/metrics/text_quality.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/metrics/text_quality.py
```

## See also

- {doc}`/cookbook/metrics/text/text_pii_safety`: PII leak rates over a text column.
- {doc}`/cookbook/metrics/text/text_retrieval`: is the answer actually supported by the retrieved context?
- {doc}`/ml/evaluation/evaluation`: scoring a model, per segment, in one pass.
- {doc}`/api/models/metrics`: the complete metric vocabulary.
