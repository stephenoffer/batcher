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
