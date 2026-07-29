# Degeneracy detection: repetition, truncation, refusal, and empty output

A model that has started looping produces text that is long and nearly information-free. The character n-gram measures catch that reliably; ``truncation_rate`` and ``refusal_rate`` catch the two other common failure shapes.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/metrics/text_diversity.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/metrics/text_diversity.py
```
