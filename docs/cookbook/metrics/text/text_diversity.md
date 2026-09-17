# Degeneracy detection

A model that has started looping produces text that is long and nearly information-free. The character n-gram measures catch that reliably.

The script compares a looping `the the the` output against a healthy sentence: the distinct n-gram ratio collapses while the repetition rate and compression proxy spike. `truncation_rate`, `refusal_rate`, and `empty_generation_rate` catch the other common failure shapes.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/metrics/text_diversity.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/metrics/text_diversity.py
```

## See also

- {doc}`/cookbook/metrics/text/text_quality`: what fraction of a text column looks broken.
- {doc}`/cookbook/metrics/text/text_formatting`: did the model obey the output format you asked for?
- {doc}`/ml/evaluation/evaluation`: scoring a model, per segment, in one pass.
- {doc}`/api/models/metrics`: the complete metric vocabulary.
