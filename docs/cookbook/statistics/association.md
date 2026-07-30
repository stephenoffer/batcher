# How strongly does one column relate to another?

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

- {doc}`ab_test_inference`: A/B test statistics computed in the engine: effect size, t-statistic, and intervals.
- {doc}`distribution_shape`: is this column symmetric, skewed, or heavy-tailed?
- {doc}`../../ml/statistics-and-drift`: the statistics surface in full, with drift and validation.
- {doc}`../../api/ml-statistics`: the reference for every statistical function.
