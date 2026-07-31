# Degeneracy detection

A model that has started looping produces text that is long and nearly information-free. The character n-gram measures catch that reliably; ``truncation_rate`` and ``refusal_rate`` catch the two other common failure shapes.

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

- {doc}`/cookbook/metrics/model/regression_errors`: absolute, squared, percentage, and robust.
- {doc}`/cookbook/metrics/text/text_formatting`: did the model obey the output format you asked for?
- {doc}`/ml/evaluation/evaluation`: scoring a model, per segment, in one pass.
- {doc}`/api/models/metrics`: the complete metric vocabulary.
