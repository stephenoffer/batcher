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
