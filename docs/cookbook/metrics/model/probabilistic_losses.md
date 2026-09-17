# Probabilistic losses

A classifier that says 0.51 and one that says 0.99 both predict the positive class, but they are not equally right. Log loss, Brier score, and the hinge losses read the score column, which is how you tell a confident model from a lucky one.

The script scores a confident model and a timid one with identical hard labels and asserts that both losses prefer the confident one. It ends with the Poisson, gamma, and Tweedie deviances for count and cost targets.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/metrics/probabilistic_losses.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/metrics/probabilistic_losses.py
```

## See also

- {doc}`/cookbook/metrics/model/classification`: classification metrics computed as aggregates over a predictions table.
- {doc}`/cookbook/metrics/model/regression_errors`: absolute, squared, percentage, and robust.
- {doc}`/ml/evaluation/evaluation`: scoring a model, per segment, in one pass.
- {doc}`/api/models/metrics`: the complete metric vocabulary.
