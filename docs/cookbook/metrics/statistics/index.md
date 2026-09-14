# Statistics cookbook

Six recipes, ordered as a reading of one column. Summarize it, measure its spread, look at its shape, then relate it to something else. Every one is an aggregate the engine evaluates, so a summary over a billion rows is one pass rather than a pull into pandas.

Each page embeds a complete, self-contained script from [`examples/statistics/`](https://github.com/stephenoffer/batcher/tree/main/examples/statistics) that builds its own in-memory data and asserts on its own output, so a page that stops matching the engine fails the suite instead of drifting.

| Recipe | What it shows |
|---|---|
| {doc}`/cookbook/metrics/statistics/summary_statistics` | Summary aggregates beyond mean and stddev |
| {doc}`/cookbook/metrics/statistics/quantiles_and_histograms` | Quantiles, histograms, and the exact-versus-approximate trade |
| {doc}`/cookbook/metrics/statistics/robust_dispersion` | Quantile-based spread that one outlier cannot move |
| {doc}`/cookbook/metrics/statistics/distribution_shape` | Whether a column is symmetric, skewed, or heavy-tailed |
| {doc}`/cookbook/metrics/statistics/association` | How strongly one column relates to another |
| {doc}`/cookbook/metrics/statistics/ab_test_inference` | Effect size, t-statistic, and confidence intervals |

## See also

- {doc}`/ml/evaluation/statistics-and-drift`: the same aggregates applied to drift monitoring.
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
