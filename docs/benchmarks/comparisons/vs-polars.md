# vs Polars

This page compares Batcher with Polars on single-node analytics and on GPU aggregation.

Batcher is faster than Polars on every suite on the current board, by 1.9x to 2.7x on TPC-H, ClickBench and the H2O.ai tasks and by far more on JSON and the operator mix. The widest single margins are in sorting and windows, where a fused top-N is 50x and `lag()` is 17x.

Every timing passed the correctness gate first. The harness runs Polars through its native `LazyFrame` pipelines, the way Polars' own TPC-H benchmark does.

## The suites

Swept 2026-09-13 on a 48-core (24 physical plus SMT), 92 GiB box, best of five, one process per case. Each cell is a suite geomean of `batcher_ms / polars_ms`, so **below 1.00 means Batcher is faster**:

| Suite | vs Polars |
|---|---:|
| Semi-structured JSON (5) | **0.01** |
| Operator mix (46) | **0.16** |
| ClickBench (43) | **0.37** |
| H2O.ai `join` (5) | **0.51** |
| H2O.ai `groupby` (10) | **0.53** |
| TPC-H sf1 (22) | **0.54** |

At ten times the data the lead holds. The five-engine board of 2026-08-28, on a 92-core box, measured TPC-H sf10 at **0.35x** against Polars.

## Operators

The operator mix times single kernels over TPC-H `lineitem` at sf1 (6,001,215 rows), read once into Arrow and shared byte-identically. The following table is the complete run of 2026-07-13 on a 16-core, 30 GB node, from before the mix grew to 46 cases. The ratio is `batcher / polars`, so **below 1.0 means Batcher is faster**:

| Operator | Batcher | Polars | vs Polars |
|---|---:|---:|---:|
| Sort then top-N | 14.1 ms | 600.7 ms | **0.02x** |
| Window `lag()` | 179.7 ms | 3,216.9 ms | **0.06x** |
| Filter then count | 0.6 ms | 8.4 ms | **0.07x** |
| Window running `sum()` | 171 ms | 786 ms | **0.22x** |
| Window `rank()` | 220.7 ms | 988.8 ms | **0.22x** |
| Global sum | 0.5 ms | 1.8 ms | **0.27x** |
| Group-by, two keys | 11.6 ms | 28.8 ms | **0.40x** |
| Group-by sum, one key | 7.6 ms | 17.1 ms | **0.44x** |
| Join then aggregate | 98.3 ms | 86.9 ms | 1.13x |
| Window `sum()` over partition | 92.7 ms | 73.8 ms | 1.26x |
| Filter then project | 13.9 ms | 9.2 ms | 1.51x |

Top-N is 50x because a fused top-N heap keeps only the running best rows and never sorts the relation. The filtered count is 14x because `.count()` over a filter compiles to a `COUNT(*)`, projection pushdown prunes the scan to the predicate's column, and the count fuses into one {py:func}`count_if <batcher.count_if>` pass.

## Join planning

Polars' TPC-H pipelines are written with the join order chosen by hand. Batcher chooses the join order itself, through a bushy dynamic-programming search over the join graph. On the 2026-07-28 board, on a c5d.24xlarge with 96 vCPU, Batcher's TPC-H geomean against Polars was **0.538x** at sf1 and **0.468x** at sf10.

The top-N and the join are both ordinary API calls. The planner fuses the first and orders the second:

```python
import batcher as bt

sales = bt.from_pydict({"store": [1, 2, 1, 3], "amount": [30, 10, 50, 20]})
stores = bt.from_pydict({"store": [1, 2, 3], "city": ["Oslo", "Lima", "Pune"]})
top = sales.join(stores, on="store").sort("amount", descending=True).limit(2)
print(top.select("city", "amount").to_pydict())
# {'city': ['Oslo', 'Oslo'], 'amount': [50, 30]}
```

## GPU aggregation

Polars' GPU engine runs on cuDF, so the like-for-like comparison is cuDF itself. The following group-by sum over 1,000 groups ran on an 8xT4 cluster ([`benchmarks/gpu_backend/distributed_cudf.py`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/gpu_backend/distributed_cudf.py)):

| Rows | Single-GPU cuDF | Batcher distributed over 8 GPUs |
|---|---:|---:|
| 200M | **1,983M rows/s** | 768M rows/s |
| 600M | OOM | **10,731M rows/s** |
| 1.2B | OOM | **13,358M rows/s** |
| 2.0B | OOM | **10,799M rows/s** |

Batcher's distributed path keeps going to 2 billion rows past one GPU's memory. Batcher uses cuDF as the per-GPU data plane rather than competing with it.

:::{dropdown} Scope of these numbers
- The operator table is from an older 16-core machine than the suite board. Compare ratios within it.
- Results are single node unless a table names a cluster.
:::

## Reproduce

```bash
python benchmarks/run.py --benchmark tpch --engines batcher,polars
python benchmarks/run.py --benchmark clickbench --engines batcher,polars
python benchmarks/run.py --benchmark operators --tier single
python benchmarks/gpu_backend/distributed_cudf.py
```

## See also

- {doc}`/benchmarks/results/analytics` for the operator table with DuckDB alongside.
- {doc}`/benchmarks/comparisons/vs-duckdb` and {doc}`/benchmarks/comparisons/vs-daft` for the other single-node comparisons.
- {doc}`/architecture/deep-dives/operators/sort-internals` for the fused top-N heap and the parallel sample sort.
- {doc}`/architecture/deep-dives/operators/window-internals` for the window kernels.
- {doc}`/architecture/deep-dives/distribution/gpu-execution` for cuDF as the per-GPU data plane.
- {doc}`/user-guide/transform/rows/sorting` for `top_k` and `sort` in the API.
- {doc}`/benchmarks/methodology` for hardware and correctness gating.
