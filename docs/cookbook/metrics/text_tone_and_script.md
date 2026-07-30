# Tone and writing-system rates: style drift and language mix

Tone rates catch a model that has become hedging or sycophantic after a prompt change. Script rates catch a corpus that is not the language you think it is, which is the usual reason a "multilingual" eval quietly measures English.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/metrics/text_tone_and_script.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/metrics/text_tone_and_script.py
```

## See also

- {doc}`text_retrieval`: RAG groundedness: is the answer actually supported by the retrieved context?
- {doc}`text_quality`: corpus hygiene rates: what fraction of a text column looks broken.
- {doc}`../../ml/evaluation`: scoring a model, per segment, in one pass.
- {doc}`../../api/metrics`: the complete metric vocabulary.
