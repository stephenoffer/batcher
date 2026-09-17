# Dataset verbs

The verbs that change a table's shape, and the calls that get a result back out of the engine. Joins and grouping carry most real pipelines, so start there. Reshaping covers pivots, explodes, and set operations, iteration covers what to call at the end, and the SQL recipe shows the same plans written as queries.

| Recipe | What it shows |
|---|---|
| {doc}`/cookbook/dataset/verbs/joins` | Join types, key spellings, and the as-of join for time series |
| {doc}`/cookbook/dataset/verbs/grouping` | `agg`, multi-key rollups, and the cube/rollup/grouping-set variants |
| {doc}`/cookbook/dataset/verbs/reshaping` | Pivot, unpivot, explode, unnest, and set operations |
| {doc}`/cookbook/dataset/verbs/iteration` | Batches, rows, slices, and the single-value cases |
| {doc}`/cookbook/dataset/verbs/sql_interface` | SQL over the same engine, and mixing SQL with DataFrame verbs |

```{toctree}
:hidden:

joins
grouping
reshaping
iteration
sql_interface
```
