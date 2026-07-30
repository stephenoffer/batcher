# Corpus hygiene rates: what fraction of a text column looks broken

Every metric here is an aggregate returning a rate in [0, 1], so one ``select`` gives you a scorecard for a whole generation run. These are the numbers you watch between model versions: a jump in ``empty_or_whitespace_rate`` is a broken prompt, not a worse model.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/metrics/text_quality.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/metrics/text_quality.py
```

## See also

- {doc}`text_pii_safety`: PII leak rates over a text column.
- {doc}`text_retrieval`: RAG groundedness: is the answer actually supported by the retrieved context?
- {doc}`../../ml/evaluation`: scoring a model, per segment, in one pass.
- {doc}`../../api/metrics`: the complete metric vocabulary.
