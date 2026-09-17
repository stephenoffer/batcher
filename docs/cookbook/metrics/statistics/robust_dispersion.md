# Robust dispersion

Standard deviation is a poor summary of a long-tailed column. One extreme row is enough to move it.

The script appends a single value of 100,000 to the numbers 1 through 20. The standard deviation grows more than a hundredfold, while the median, trimean, and midhinge barely move. Reach for these quantile-based measures, such as the interdecile range and the robust coefficient of variation, when one bad row shouldn't make a dashboard flap.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/statistics/robust_dispersion.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/statistics/robust_dispersion.py
```

## See also

- {doc}`/cookbook/metrics/statistics/quantiles_and_histograms`: quantiles, histograms, and the exact-versus-approximate trade.
- {doc}`/cookbook/metrics/statistics/summary_statistics`: summary aggregates beyond mean and stddev.
- {doc}`/ml/evaluation/statistics-and-drift`: the statistics surface in full, with drift and hypothesis tests.
- {doc}`/api/models/ml-statistics`: the reference for every statistical function.
