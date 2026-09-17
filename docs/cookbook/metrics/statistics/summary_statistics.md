# Summary statistics

The averages here answer different questions. Geometric mean is the right average for growth rates, harmonic mean for rates and speeds, and RMS for magnitudes that cancel. Using the arithmetic mean for all three is the most common quiet mistake in a metrics table.

The script also covers weighted mean and variance, standard error, the coefficient of variation, and the null and distinct rates. Its last check shows nulls dropping out of the mean while `null_rate` still counts them.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/statistics/summary_statistics.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/statistics/summary_statistics.py
```

## See also

- {doc}`/cookbook/metrics/statistics/robust_dispersion`: quantile-based measures that one outlier cannot move.
- {doc}`/cookbook/metrics/statistics/quantiles_and_histograms`: quantiles, histograms, and the exact-versus-approximate trade.
- {doc}`/ml/evaluation/statistics-and-drift`: the statistics surface in full, with drift and hypothesis tests.
- {doc}`/api/models/ml-statistics`: the reference for every statistical function.
