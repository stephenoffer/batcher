# Classification metrics

These are aggregate expressions, so evaluation is a ``select`` (or a ``group_by`` if you want the metric per segment) rather than a pull into pandas. On a table too big for memory that difference is the whole ballgame.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/metrics/classification.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/metrics/classification.py
```

## See also

- {doc}`/cookbook/metrics/model/agreement`: how well a prediction tracks the truth, not just how close.
- {doc}`/cookbook/metrics/model/diagnostic`: the epidemiology-style view of a binary classifier.
- {doc}`/ml/evaluation/evaluation`: scoring a model, per segment, in one pass.
- {doc}`/api/models/metrics`: the complete metric vocabulary.
