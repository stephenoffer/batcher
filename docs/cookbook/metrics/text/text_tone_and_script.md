# Tone and script mix

Tone rates catch a model that has become hedging or sycophantic after a prompt change. Script rates catch a corpus that is not the language you think it is, which is the usual reason a "multilingual" eval quietly measures English.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/metrics/text_tone_and_script.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/metrics/text_tone_and_script.py
```

## See also

- {doc}`/cookbook/metrics/text/text_retrieval`: is the answer actually supported by the retrieved context?
- {doc}`/cookbook/metrics/text/text_quality`: what fraction of a text column looks broken.
- {doc}`/ml/evaluation/evaluation`: scoring a model, per segment, in one pass.
- {doc}`/api/models/metrics`: the complete metric vocabulary.
