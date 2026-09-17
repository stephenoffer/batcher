# A/B test inference

An A/B test in Batcher is a set of aggregate expressions over the assignment table, so it runs where the data is instead of pulling a sample into SciPy.

The script computes Cohen's d and Hedges' g for effect size, Welch's t-statistic and degrees of freedom, confidence half-widths, and a proportion z-statistic for the conversion rate. Then it repeats the means and intervals per arm under a `group_by`. Welch's test doesn't assume equal variances, which is the right default for a real experiment.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/statistics/ab_test_inference.py
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
- {doc}`/ml/evaluation/statistics-and-drift`: the statistics surface in full, with drift and hypothesis tests.
- {doc}`/api/models/ml-statistics`: the reference for every statistical function.
