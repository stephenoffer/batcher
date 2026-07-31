# Dataset verbs

The operations that change the shape of a table, plus the two ways to get results back out.

Each page embeds a complete, self-contained script that builds its own in-memory data and asserts on its own output, so a page that stops matching the engine fails the suite instead of drifting.

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
