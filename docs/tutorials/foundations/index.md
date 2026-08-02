# Foundations

These three tutorials teach the engine itself. Start here if you have not written a Batcher
pipeline before, and read them in order: the first builds the mental model the other two
assume.

| Tutorial | What you build |
|---|---|
| {doc}`Your first pipeline <first-pipeline>` | A complete pipeline over an in-memory dataset: derive a column, group it, aggregate it, and collect |
| {doc}`From SQL to DataFrames <sql-to-dataframe>` | One SQL query, rewritten as a DataFrame chain, so the two spellings sit side by side |
| {doc}`Optimizing a slow query <optimizing-a-slow-query>` | The loop you run when a query is slow: read the plan, measure, change one thing, measure again |

## See also

- {doc}`/tutorials/pipelines/index`: the same skills applied to a lakehouse, a stream, and a generated dataset.
- {doc}`/getting-started/concepts/index`: the concepts behind what these tutorials do.

```{toctree}
:hidden:

first-pipeline
sql-to-dataframe
optimizing-a-slow-query
```
