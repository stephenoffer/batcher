# RAG groundedness: is the answer actually supported by the retrieved context?

These compare an answer column against the context column it was generated from, so they run over an existing RAG output table with no extra model call. A falling ``fully_grounded_rate`` is the signal that retrieval regressed, not generation.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/metrics/text_retrieval.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/metrics/text_retrieval.py
```

## See also

- {doc}`text_quality`: corpus hygiene rates: what fraction of a text column looks broken.
- {doc}`text_tone_and_script`: tone and writing-system rates: style drift and language mix.
- {doc}`../../ml/evaluation`: scoring a model, per segment, in one pass.
- {doc}`../../api/metrics`: the complete metric vocabulary.
