# vs PyArrow

This page compares Batcher with PyArrow's compute kernels on the work PyArrow can express.

PyArrow is Batcher's substrate as much as its competitor. Batcher's data plane speaks Arrow, and several of its kernels call into Arrow's. Everything above the buffer differs: morsel scheduling across cores, a query optimizer, and operators that stream rather than materialize. On the operator mix, that difference makes Batcher about 30x faster.

## Where a comparison exists

PyArrow has no SQL surface and no query planner. It has `Table` and `compute`, so a benchmark case can be written for it only when the case is a single operator or a short chain of them. The operator mix is that kind of suite, and TPC-H, TPC-DS, ClickBench, JSON and the H2O.ai tasks are multi-table SQL that PyArrow can't express. A blank on those suites means PyArrow can't run the query, not that it lost.

## The measured standing

On the five-engine board of 2026-08-28, on a 92-core box with best of five and every case correctness-gated against DuckDB, Batcher's operator-mix geomean against PyArrow was **0.03x**. The ratio is `batcher_ms / pyarrow_ms`, so lower is better.

A sweep of 2026-07-20 on a 16-core node recorded the operators one by one, with the same convention:

| Operator | vs PyArrow |
|---|---:|
| Filter then count | **0.00x** |
| Sort then top-N | **0.01x** |
| Global sum | **0.04x** |
| Filter then project | **0.06x** |
| Join then aggregate | **0.32x** |
| Group-by, two keys | **0.54x** |
| Group-by sum, one key | **0.60x** |

String kernels show the same pattern. On the 2026-09-12 operator additions, a `LENGTH` over a string column took 10.9 ms in Batcher against 122.1 ms for single-threaded PyArrow, and a `LIKE '%...%'` 13.4 ms against 482.9 ms.

## Why the margin is wide

This isn't kernel against kernel. PyArrow executes an operator over a whole table on one thread. Batcher splits the same work into morsels of 16,384 rows or 1 MiB and runs them across every core. Where PyArrow must materialize an intermediate, such as a sort feeding a limit, Batcher's fused operators never allocate it: a top-N keeps only the running best rows. And a filtered count reads one column, because the optimizer prunes the scan to the predicate before anything decodes.

## Requirements and limitations

The following limits apply to this comparison:

- **Kernel for kernel, the engines converge.** On a single vectorized pass over one column the two are the same order, and several of Batcher's kernels are Arrow's. The result is a statement about scheduling and planning, not about faster kernels.
- **Coverage is the operator mix only.** Multi-table suites have no PyArrow column.

## Reproduce

The following command reruns the operator mix with PyArrow in the lineup:

```bash
python benchmarks/run.py --benchmark operators --engines batcher,duckdb,pyarrow
```

## See also

- {doc}`/benchmarks/results/engine-matrix`: PyArrow beside every other engine.
- {doc}`/benchmarks/methodology`: the correctness gate and the hardware.
- {doc}`/architecture/deep-dives/operators/morsel-parallelism`: how the morsel scheduler sits over the Arrow kernels.
- {doc}`/architecture/index`: the engine's architecture as a whole.
