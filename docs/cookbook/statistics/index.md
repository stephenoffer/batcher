# Statistics cookbook

Summary statistics, robust dispersion, distribution shape, association, and A/B test inference, all as aggregates in the engine.

Every page here embeds a complete, self-contained script from the
[`examples/statistics/`](https://github.com/batcher/batcher/tree/main/examples/statistics) directory.
The scripts build their own in-memory data and assert on their own output, and
`tests/docs/test_examples.py` runs all of them, so a page that stops matching the
engine fails the suite instead of drifting.

| Recipe | What it shows |
|---|---|
| {doc}`ab_test_inference` | A/B test statistics computed in the engine: effect size, t-statistic, and intervals |
| {doc}`association` | How strongly does one column relate to another? |
| {doc}`distribution_shape` | Is this column symmetric, skewed, or heavy-tailed? |
| {doc}`quantiles_and_histograms` | Quantiles, histograms, and the exact-versus-approximate trade |
| {doc}`robust_dispersion` | Robust spread: quantile-based measures that one outlier cannot move |
| {doc}`summary_statistics` | Summary aggregates beyond mean and stddev |

```{toctree}
:hidden:

ab_test_inference
association
distribution_shape
quantiles_and_histograms
robust_dispersion
summary_statistics
```
