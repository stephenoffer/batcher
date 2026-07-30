# Diagnostic metrics: the epidemiology-style view of a binary classifier

Accuracy hides everything on an imbalanced problem. Likelihood ratios, informedness, and markedness describe how much a prediction actually moves your belief, which is the number you want when positives are rare.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/metrics/diagnostic.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/metrics/diagnostic.py
```

## See also

- {doc}`classification`: classification metrics computed as aggregates over a predictions table.
- {doc}`embeddings`: corpus-level embedding metrics: monitoring a vector column in aggregate.
- {doc}`../../ml/evaluation`: scoring a model, per segment, in one pass.
- {doc}`../../api/metrics`: the complete metric vocabulary.
