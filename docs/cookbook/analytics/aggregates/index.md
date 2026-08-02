# Aggregates and rankings

These recipes are the shapes most analytical queries actually are: group, join, window, and
rank. Start from the worked query if you want the whole shape in one page.

| Recipe | The question |
|---|---|
| {doc}`Analytics query <analytics-query>` | Aggregate, join, then window, in one readable pass |
| {doc}`Time series rollups <time-series-rollups>` | Daily revenue with a seven-day moving average |
| {doc}`Top k per group <top-k-per-group>` | The two best sellers in every category, not the two best sellers overall |

## See also

- {doc}`/user-guide/analyze/aggregations`: the aggregate reference behind these recipes.
- {doc}`/cookbook/analytics/behavior/index`: the same table, asked about people rather than totals.

```{toctree}
:hidden:

analytics-query
time-series-rollups
top-k-per-group
```
