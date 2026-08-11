# Head to head

This section holds one page per competing engine, with the measured standing against each
and the architectural reason behind it.

Read these when you are choosing between Batcher and something you already run. Read
{doc}`/benchmarks/results/index` instead when you want the standing on a workload rather
than against a name.

| Page | The short version |
|---|---|
| {doc}`vs DuckDB <vs-duckdb>` | Batcher takes the operator mix and the shared-Arrow suites |
| {doc}`vs Polars <vs-polars>` | Batcher takes sort, top-N, and windows by a wide margin, and the sf10 suite by 2.26x |
| {doc}`vs Daft <vs-daft>` | Batcher takes image decode, top-N, and the distributed join, and is correct where Daft is not |
| {doc}`vs Spark <vs-spark>` | An architectural comparison: where each engine re-plans a query, and what that costs |

:::{note}
Every comparison runs each engine over the identical zero-copy Arrow input unless the page
says otherwise, so it measures execution rather than storage format. Where a comparison is
deliberately not like-for-like, such as DuckDB reading its own compressed store, the page
says so and publishes both columns.
:::

## See also

- {doc}`/benchmarks/methodology`: the correctness gate every one of these numbers passed first.
- {doc}`/getting-started/migration/index`: porting a workload off one of these engines, verb by verb.

```{toctree}
:hidden:

vs-duckdb
vs-polars
vs-daft
vs-spark
```
