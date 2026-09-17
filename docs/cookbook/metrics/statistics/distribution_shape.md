# Distribution shape

Shape decides which summary is honest. On a skewed column the mean is not the typical value, and a test that assumes normality is not valid.

The script runs moment and quantile skewness, kurtosis, and the Jarque-Bera statistic over a symmetric column and a right-skewed one, then confirms the skewed column's mean sits above its median. Run these before you pick the summary, not after someone questions the dashboard.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/statistics/distribution_shape.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/statistics/distribution_shape.py
```

## See also

- {doc}`/cookbook/metrics/statistics/association`: how strongly does one column relate to another?
- {doc}`/cookbook/metrics/statistics/quantiles_and_histograms`: quantiles, histograms, and the exact-versus-approximate trade.
- {doc}`/ml/evaluation/statistics-and-drift`: the statistics surface in full, with drift and hypothesis tests.
- {doc}`/api/models/ml-statistics`: the reference for every statistical function.
