# Head to head

This section compares Batcher with one engine per page, each with the measured standing and the architectural reason behind it. Read these pages when you are choosing between Batcher and something you already run.

For the standing on a kind of workload rather than against a named engine, {doc}`/benchmarks/results/index` arranges the same numbers that way.

The following table summarizes each page:

| Page | The short version |
|---|---|
| {doc}`vs DuckDB <vs-duckdb>` | Faster on every suite over the same Arrow, and on five of six against DuckDB's own compressed store |
| {doc}`vs Polars <vs-polars>` | Faster on every suite on the current board, and 50x on top-N |
| {doc}`vs Daft <vs-daft>` | Faster on every analytical suite, about 2x on image decode, and 1.7x to 2.2x on the distributed join |
| {doc}`vs Spark <vs-spark>` | 20x to 50x on TPC-H sf1 on one node, and how the two architectures differ |
| {doc}`vs PyArrow <vs-pyarrow>` | The same Arrow kernels underneath, with a scheduler and a planner that make the operator mix about 30x faster |

:::{note}
Each page states its ratio convention in every table's lead-in. Suite tables report `batcher_ms / engine_ms`, where lower is better, and the distributed tables report `engine_ms / batcher_ms`, where higher is better. The two read identically and mean opposite things.
:::

## See also

- {doc}`/benchmarks/methodology`: the correctness gate every one of these numbers passed first.
- {doc}`/benchmarks/results/engine-matrix`: every engine on one board.
- {doc}`/getting-started/migration/index`: porting a workload off one of these engines, verb by verb.

```{toctree}
:hidden:

vs-duckdb
vs-polars
vs-daft
vs-spark
vs-pyarrow
```
