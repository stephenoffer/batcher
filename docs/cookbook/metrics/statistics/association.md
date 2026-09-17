# Association

Correlation is for two numeric columns. When one side is a binary outcome, such as churn, the point-biserial correlation and the signal ratio are the measures built for it.

The script covers both cases, adds weighted correlation and covariance for rows of unequal importance, and fits an ordinary least-squares line with the `regr_*` aggregates. It all runs as aggregates, with no model object and no pull into Python.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/statistics/association.py
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
- {doc}`/ml/evaluation/statistics-and-drift`: the statistics surface in full, with drift and hypothesis tests.
- {doc}`/api/models/ml-statistics`: the reference for every statistical function.
