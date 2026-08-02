# Statistics cookbook

This section holds 6 runnable recipes that compute statistics as aggregates in the engine, so a summary over a billion rows is one pass rather than a pull into pandas.

They are ordered as a reading of a column: summarize it, measure its spread, look at its shape, then relate it to something else.

Every page embeds a complete, self-contained script from the [`examples/statistics/`](https://github.com/batcher/batcher/tree/main/examples/statistics) directory. The scripts build their own in-memory data and assert on their own output, and `tests/docs/test_examples.py` runs all of them, so a page that stops matching the engine fails the suite instead of drifting.

| Recipe | What it shows |
|---|---|
| {doc}`/cookbook/metrics/statistics/summary_statistics` | Summary aggregates beyond mean and stddev |
| {doc}`/cookbook/metrics/statistics/quantiles_and_histograms` | Quantiles, histograms, and the exact-versus-approximate trade |
| {doc}`/cookbook/metrics/statistics/robust_dispersion` | Quantile-based spread that one outlier cannot move |
| {doc}`/cookbook/metrics/statistics/distribution_shape` | Whether a column is symmetric, skewed, or heavy-tailed |
| {doc}`/cookbook/metrics/statistics/association` | How strongly one column relates to another |
| {doc}`/cookbook/metrics/statistics/ab_test_inference` | Effect size, t-statistic, and confidence intervals |

## See also

- {doc}`/ml/evaluation/statistics-and-drift`: the same aggregates applied to drift and honest splits.
- {doc}`/api/models/ml-statistics`: the statistical function reference.
- {doc}`/cookbook/dataset/inspecting/profiling`: the first pass over an unfamiliar table.
- {doc}`/cookbook/metrics/index`: scoring a model, as opposed to describing a column.

```{toctree}
:hidden:

summary_statistics
quantiles_and_histograms
robust_dispersion
distribution_shape
association
ab_test_inference
```
