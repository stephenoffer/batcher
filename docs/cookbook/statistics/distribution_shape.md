# Distribution shape

Shape decides which summary is honest. On a skewed column the mean is not the typical value, and a normality-assuming test is not valid. These aggregates answer that question before you pick the summary rather than after someone questions the dashboard.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/statistics/distribution_shape.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/statistics/distribution_shape.py
```

## See also

- {doc}`association`: how strongly does one column relate to another?
- {doc}`quantiles_and_histograms`: quantiles, histograms, and the exact-versus-approximate trade.
- {doc}`/ml/evaluation/statistics-and-drift`: the statistics surface in full, with drift and validation.
- {doc}`/api/models/ml-statistics`: the reference for every statistical function.
