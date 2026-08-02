# Drawing conclusions

These recipes go past describing the data to deciding something from it: which products go
together, whether a change actually helped, and which row is not like the others. Each one
is an aggregate rather than a pull into a statistics library.

| Recipe | The question |
|---|---|
| {doc}`Basket analysis <basket-analysis>` | Which products get bought together, via a self-join |
| {doc}`A/B testing <ab-testing>` | Did variant B convert better than A, with an interval rather than a point |
| {doc}`Anomaly detection <anomaly-detection>` | Which host in the fleet started answering slowly |

## See also

- {doc}`/cookbook/metrics/statistics/index`: the statistical aggregates these are built from.
- {doc}`/ml/evaluation/statistics-and-drift`: the same machinery applied to model drift.

```{toctree}
:hidden:

basket-analysis
ab-testing
anomaly-detection
```
