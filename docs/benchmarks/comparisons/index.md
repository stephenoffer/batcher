# Head to head

One page per competing engine, each with the measured standing and the architectural reason
behind it. Read them when you are choosing between Batcher and something you already run.

Want the standing on a workload rather than against a name? {doc}`/benchmarks/results/index`
is arranged that way instead.

Every page below states its own ratio convention in each table's lead-in, because the
scorecard tables report speedups and the operator tables report `batcher / competitor` times.
The two read identically and mean opposite things.

| Page | The short version |
|---|---|
| {doc}`vs DuckDB <vs-duckdb>` | Batcher takes the operator mix and the shared-Arrow suites |
| {doc}`vs Polars <vs-polars>` | Batcher takes sort, top-N, and windows by a wide margin, and the sf10 suite by 2.26x |
| {doc}`vs Daft <vs-daft>` | Batcher takes image decode, top-N, and the distributed join, and is correct where Daft is not |
| {doc}`vs Spark <vs-spark>` | An architectural comparison, and the measured board: Batcher takes every suite |
| {doc}`vs PyArrow <vs-pyarrow>` | Same Arrow kernels underneath; the margin is scheduling and planning |

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
vs-pyarrow
```
