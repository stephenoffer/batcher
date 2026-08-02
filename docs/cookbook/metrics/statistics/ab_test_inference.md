# A/B test inference

The whole test is aggregate expressions over the assignment table, so it runs where the data is instead of pulling a sample into SciPy. ``welch_*`` does not assume equal variances, which is the right default for a real experiment.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/statistics/ab_test_inference.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/statistics/ab_test_inference.py
```

## See also

- {doc}`/cookbook/metrics/statistics/association`: how strongly does one column relate to another?
- {doc}`/cookbook/metrics/statistics/distribution_shape`: is this column symmetric, skewed, or heavy-tailed?
- {doc}`/ml/evaluation/statistics-and-drift`: the statistics surface in full, with drift and validation.
- {doc}`/api/models/ml-statistics`: the reference for every statistical function.
