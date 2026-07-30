# Robust spread: quantile-based measures that one outlier cannot move

Standard deviation is a poor summary of a long-tailed distribution, which describes most latency and revenue data. These are the quantile-based alternatives: a single extreme row shifts them barely at all, so a dashboard built on them stops flapping.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/statistics/robust_dispersion.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/statistics/robust_dispersion.py
```

## See also

- {doc}`quantiles_and_histograms`: quantiles, histograms, and the exact-versus-approximate trade.
- {doc}`summary_statistics`: summary aggregates beyond mean and stddev.
- {doc}`../../ml/statistics-and-drift`: the statistics surface in full, with drift and validation.
- {doc}`../../api/ml-statistics`: the reference for every statistical function.
