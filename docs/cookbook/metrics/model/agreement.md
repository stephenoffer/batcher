# Agreement metrics

Correlation says the shapes match. Concordance correlation, Nash-Sutcliffe efficiency, and Kling-Gupta efficiency say the *values* match.

The script scores two forecasts against the same observations: a close tracker, and a series that is perfectly correlated but shifted up by 10. Plain correlation gives the shifted series 1.0. Every agreement metric marks it down, and Nash-Sutcliffe goes negative, because the forecast does worse than predicting the mean.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/metrics/agreement.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/metrics/agreement.py
```

## See also

- {doc}`/cookbook/metrics/model/classification`: classification metrics computed as aggregates over a predictions table.
- {doc}`/cookbook/metrics/model/diagnostic`: the epidemiology-style view of a binary classifier.
- {doc}`/ml/evaluation/evaluation`: scoring a model, per segment, in one pass.
- {doc}`/api/models/metrics`: the complete metric vocabulary.
