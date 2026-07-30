# Agreement metrics: how well a prediction tracks the truth, not just how close

Correlation says the shapes match; these say the *values* match. A forecast that is perfectly correlated but biased high scores well on correlation and badly here, which is usually the honest answer.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/metrics/agreement.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/metrics/agreement.py
```

## See also

- {doc}`classification`: classification metrics computed as aggregates over a predictions table.
- {doc}`diagnostic`: diagnostic metrics: the epidemiology-style view of a binary classifier.
- {doc}`../../ml/evaluation`: scoring a model, per segment, in one pass.
- {doc}`../../api/metrics`: the complete metric vocabulary.
