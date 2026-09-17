# vs Polars

This page compares Batcher with Polars on single-node analytics and on GPU aggregation.

Batcher is faster than Polars on every suite on the current board, by 1.9x to 2.7x on TPC-H, ClickBench and the H2O.ai tasks and by far more on JSON and the operator mix. The widest single margins are in sorting and windows, where a fused top-N is 50x and `lag()` is 17x.

:::{important}
Every timing passed the correctness gate first. Polars' SQL frontend returns the wrong revenue on TPC-H q6, folding the bound `0.06 + 0.01` to `0.06999999999999999` in IEEE double and dropping every `l_discount = 0.07` row, and it fails 9 of the 22 queries outright. So the harness runs Polars through its native `LazyFrame` pipelines, the way Polars' own TPC-H benchmark does, and every TPC-H figure here is measured that way.
:::

## The suites

The current board was swept 2026-09-13 on a quiet 48-core (24 physical plus SMT), 92 GiB box, best of five, one process per case. Each cell is a suite geomean of `batcher_ms / polars_ms`, so **below 1.00 means Batcher is faster**:

| Suite | vs Polars |
|---|---:|
| Semi-structured JSON (5) | **0.01** |
| Operator mix (46) | **0.16** |
| ClickBench (43) | **0.37** |
| H2O.ai `join` (5) | **0.51** |
| H2O.ai `groupby` (10) | **0.53** |
| TPC-H sf1 (22) | **0.54** |

At ten times the data the lead holds. The five-engine board of 2026-08-28, on a 92-core box, measured TPC-H sf10 at **0.35x** against Polars.

Polars can't run TPC-DS at all. Its SQL frontend rejects the comma joins the suite is written in, with `multiple tables in FROM clause are not currently supported`.

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

Top-N is 50x because a fused top-N heap keeps only the running best rows and never sorts the relation, where Polars sorts and then takes the first ten. The filtered count is 14x because `.count()` over a filter compiles to a `COUNT(*)`, projection pushdown prunes the scan to the predicate's column, and the count fuses into one {py:func}`count_if <batcher.count_if>` pass.

## Join planning

Polars' TPC-H pipelines are written with the join order chosen by hand. Batcher takes the SQL and chooses the join order itself, through a bushy dynamic-programming search over the join graph, so the same query text plans well without an author ordering it. On the 2026-07-28 board, on a c5d.24xlarge with 96 vCPU, Batcher's TPC-H geomean against Polars was 0.538x at sf1 and 0.468x at sf10.

## GPU aggregation

Polars' GPU engine runs on cuDF, so the like-for-like comparison is cuDF itself. The following group-by sum over 1,000 groups ran on an 8xT4 cluster (`benchmarks/gpu_backend/distributed_cudf.py`):

| Rows | Single-GPU cuDF | Batcher distributed over 8 GPUs |
|---|---:|---:|
| 200M | **1,983M rows/s** | 768M rows/s |
| 600M | OOM | **10,731M rows/s** |
| 1.2B | OOM | **13,358M rows/s** |
| 2.0B | OOM | **10,799M rows/s** |

For data that fits one GPU, single-GPU cuDF is faster, because the cross-device combine isn't free. Past one GPU's memory it stops running, and Batcher's distributed path keeps going to 2 billion rows. That is a distribution result rather than a kernel result, and it is why Batcher uses cuDF as the per-GPU data plane rather than competing with it.

## Requirements and limitations

The following shapes are where Polars leads or where a figure needs its context:

- **Filter then project, join then aggregate, and a windowed `sum()` over partitions** ran faster on Polars in the operator table above, by 1.51x, 1.13x and 1.26x.
- **High-cardinality grouping and full float sorts** trailed Polars in an in-memory microbenchmark of 2026-06-28 on 16 cores, whose driver script the record doesn't name: a 1.25M-group group-by at 2.1x and a 2M-row `ORDER BY` on a float key at 1.7x, after fixes that had halved both gaps.
- **Exact `MEDIAN` and `QUANTILE_CONT` per group** trailed Polars in the same microbenchmark, 210 ms against 66 ms, because an exact median must hold every value of the group.
- **The operator table** is from a smaller, older machine than the suite board. Compare ratios within it.

## Reproduce

The following commands rerun each result:

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
