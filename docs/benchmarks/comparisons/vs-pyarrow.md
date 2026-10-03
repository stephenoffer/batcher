# vs PyArrow

This page compares Batcher with PyArrow's compute kernels on the work PyArrow can express.

PyArrow is Batcher's substrate as much as its competitor. Batcher's data plane speaks Arrow, and several of its kernels call into Arrow's. Everything above the buffer differs: morsel scheduling across cores, a query optimizer, and operators that stream rather than materialize. On the operator mix, that difference makes Batcher about 30x faster.

## Where a comparison exists

PyArrow has `Table` and `compute` but no SQL surface or query planner, so the comparison covers the operator mix only.

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

This isn't kernel against kernel. PyArrow executes an operator over a whole table on one thread. Batcher splits the same work into morsels of 16,384 rows or 1 MiB and runs them across every core. Fused operators skip intermediates, so a top-N keeps only the running best rows, and a filtered count reads one column because the optimizer prunes the scan to the predicate.

Arrow data moves in and out without a copy:

```python
import batcher as bt
import pyarrow as pa

table = pa.table({"k": ["a", "b", "a"], "v": [1, 2, 3]})
out = bt.from_arrow(table).group_by("k").agg(total=bt.col("v").sum()).sort("k").to_arrow()
print(out.to_pydict())
# {'k': ['a', 'b'], 'total': [4, 2]}
```

Several of Batcher's kernels are Arrow's, so the margin is a statement about scheduling and planning rather than faster kernels.

## Reproduce

```bash
python benchmarks/run.py --benchmark operators --engines batcher,duckdb,pyarrow
```

## See also

- {doc}`/benchmarks/results/engine-matrix`: PyArrow beside every other engine.
- {doc}`/benchmarks/methodology`: the correctness gate and the hardware.
- {doc}`/architecture/deep-dives/operators/morsel-parallelism`: how the morsel scheduler sits over the Arrow kernels.
- {doc}`/architecture/index`: the engine's architecture as a whole.
