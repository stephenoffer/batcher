# RAG groundedness

These compare an answer column against the context column it was generated from, so they run over an existing RAG output table with no extra model call. A falling ``fully_grounded_rate`` is the signal that retrieval regressed, not generation.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/metrics/text_retrieval.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/metrics/text_retrieval.py
```

## See also

- {doc}`/cookbook/metrics/text/text_quality`: what fraction of a text column looks broken.
- {doc}`/cookbook/metrics/text/text_tone_and_script`: style drift and language mix.
- {doc}`/ml/evaluation/evaluation`: scoring a model, per segment, in one pass.
- {doc}`/api/models/metrics`: the complete metric vocabulary.
