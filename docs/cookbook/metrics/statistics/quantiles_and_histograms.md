# Quantiles and histograms

Exact quantiles need the whole column ordered. Sketch-backed ones, `approx_quantile` and `approx_median`, answer within a bounded error from a fixed amount of memory, which makes them usable on a column that doesn't fit in memory. Know which one you are getting.

The script compares the two on 1,000 latency values, computes p50 and p95 per route in one `group_by`, and builds a histogram by grouping on a bucket expression.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/statistics/quantiles_and_histograms.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/statistics/quantiles_and_histograms.py
```

## See also

- {doc}`/cookbook/metrics/statistics/distribution_shape`: is this column symmetric, skewed, or heavy-tailed?
- {doc}`/cookbook/metrics/statistics/robust_dispersion`: quantile-based measures that one outlier cannot move.
- {doc}`/ml/evaluation/statistics-and-drift`: the statistics surface in full, with drift and hypothesis tests.
- {doc}`/api/models/ml-statistics`: the reference for every statistical function.
