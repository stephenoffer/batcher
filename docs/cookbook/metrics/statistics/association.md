# Association

Correlation is for two numeric columns. When one side is a category or a binary outcome you need a different measure, and reaching for Pearson anyway is how a "no signal" result gets reported on a variable that clearly has signal.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/statistics/association.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/statistics/association.py
```

## See also

- {doc}`/cookbook/metrics/statistics/ab_test_inference`: effect size, t-statistic, and intervals.
- {doc}`/cookbook/metrics/statistics/distribution_shape`: is this column symmetric, skewed, or heavy-tailed?
- {doc}`/ml/evaluation/statistics-and-drift`: the statistics surface in full, with drift and validation.
- {doc}`/api/models/ml-statistics`: the reference for every statistical function.
