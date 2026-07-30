# vs Polars

This page compares Batcher against Polars on single-node analytics.

The split is unusually sharp. Batcher takes sorting, top-N, filtered counts, and most of the window family by very large margins, up to 50x. Polars takes the high-cardinality hash paths and the exact quantiles, some of them by 2x to 3x. Which engine is faster depends on the shape of your query, so both directions are laid out below.

:::{important}
Polars' TPC-H q6 returns the **wrong revenue**, folding the bound `0.06 + 0.01` to `0.06999999999999999` in IEEE double, which drops every `l_discount = 0.07` row, and the
harness declines to time a query whose result does not match the oracle. So Polars has no
number on q6 rather than a fast one. That gate runs before every timing on this page, in
both directions.
:::

## Scorecard

Each row is one workload shape, with the engine that won it and by how much. Read them
against the methodology above rather than in isolation:

| Shape | Winner |
|---|---|
| Sort → top-N (`LIMIT`) | Batcher, 50× |
| Window `lag()` | Batcher, 17× |
| Window running `sum()` | Batcher, 4.6× |
| Filter → count | Batcher, 14× |
| Group-by (low cardinality) | Batcher, 2.3× |
| Group-by (high cardinality, 1.25M groups) | Polars, 2.1× |
| Full sort on a float key | Polars, 1.7× |
| `SUM() OVER (PARTITION BY)` | Polars, 3.1× |
| TPC-H overall (sf10), same Arrow input | Batcher, 2.26×; wins 17 of 22 |
| `MEDIAN` / `QUANTILE_CONT` per group | Polars, 210 ms vs 66 ms |
| TPC-H through the SQL frontend | Polars errors on most of the suite |

## Operators

Single node, 16 cores, 30 GB. TPC-H `lineitem` at scale factor 1 (6,001,215 rows), read
once into Arrow and shared byte-identically. The ratio is `batcher / polars`, so **below
1.0 means Batcher is faster**.

| Operator | Batcher | Polars | vs Polars |
|---|---:|---:|---:|
| sort → top-N (`LIMIT`) | 14.1 ms | 601 ms | **0.02×** |
| window `lag()` | 180 ms | 3,217 ms | **0.06×** |
| filter → count | 0.6 ms | 8.4 ms | **0.07×** |
| window running `sum()` | 171 ms | 786 ms | **0.22×** |
| window `rank()` | 221 ms | 989 ms | **0.22×** |
| global sum | 0.5 ms | 1.8 ms | **0.27×** |
| group-by, two keys | 11.6 ms | 28.8 ms | **0.40×** |
| group-by sum, one key | 7.6 ms | 17.1 ms | **0.44×** |
| filter → project | 13.9 ms | 9.2 ms | 1.51× |
| join → aggregate | 98.3 ms | 86.9 ms | 1.13× |
| window `sum()` over partition | 92.7 ms | 73.8 ms | 1.26× |

Two of these deserve a sentence. Top-N is 50x because a fused top-N heap keeps the running best *k* rows and never sorts the relation, where Polars sorts and then takes ten. `lag()` is 17x because Polars' window path is slow, which the whole window column shows.

## Where Polars wins

:::{note}
Polars leads on the hash-heavy and quantile paths: **2.1x on a high-cardinality group-by**, **1.7x on a full float sort**, and **3.1x on a partitioned window `SUM`**, and an exact per-group median takes Batcher 210 ms against Polars' 66 ms. If your query is a wide hash aggregation on one node, that is the shape to measure before you choose.
:::

All on 16 cores, in-memory Arrow, correctness-gated against DuckDB *and* Polars.

| Operator | Batcher | Polars |
|---|---:|---:|
| group-by, high cardinality (5M rows → 1.25M groups) | 182 ms | 81 ms |
| `DISTINCT` (1.25M distinct ints, 5M rows) | 111 ms | 81 ms |
| full sort `ORDER BY <f64>` (2M rows) | 68 ms | 33 ms |
| two-key sort `ORDER BY <i64>, <i64>` (2M rows) | 91 ms | 65 ms |
| `SUM(x) OVER (PARTITION BY g)` (2M rows) | 85 ms | 27 ms |
| `MEDIAN(x) GROUP BY flag` (5M rows) | 210 ms | 66 ms |
| `COUNT(DISTINCT id) GROUP BY flag` (2M rows) | 163 ms | 42 ms |

Every one of those is better than it was, and the direction of travel is steep. The high-cardinality group-by ran at 400 ms before the radix combine hashed native keys directly and merged its partitions in parallel. `DISTINCT` was 300 ms, the float sort was 164 ms before sample-sort, and the two-key sort was 561 ms and single-threaded. Measured against Polars, those four gaps went from 4.7x to 2.1x, from 1.6x to 1.4x, from 4.9x to 1.7x, and from 8.9x to 1.4x.

The residual on median is the exact value-list materialization, because an exact median must hold every value, plus a three-group parallelism ceiling. The residual on the partitioned window is the executor materializing the full input ahead of the operator, not the kernel.

## SQL

Polars can't parse most of the TPC-H suite through its SQL frontend, including multi-table `FROM`, `EXISTS`, and non-equi joins, so its column in the results is mostly `ERR`. That says something about its SQL surface and nothing about its speed, and it's reported that way.

Where it does parse q6, it computes the **wrong revenue**, folding the bound `0.06 + 0.01` to `0.06999999999999999` in IEEE double, which drops every `l_discount = 0.07` row.
A wrong answer gets no timing, so Polars gets no number there rather than a fast one.
Batcher matches DuckDB on all 22 queries.

## GPU

:::{note}
The table below was measured on an 8xT4 cluster. Every table above it was measured on a single 16-core node. Those are different machines running different work, so a row from one can't be set against a row from the other. See {doc}`methodology`.
:::

Polars' GPU mode runs on cuDF, so the honest comparison is against cuDF itself. On a group-by sum over 1000 groups on an 8xT4 cluster, single-GPU cuDF is genuinely fast. For data that fits one GPU it beats Batcher's distributed cuDF path, because the cross-device combine isn't free:

| Rows | Single-GPU cuDF | Batcher distributed over 8 GPUs |
|---|---:|---:|
| 200M | **1,983 M rows/s** | 768 M rows/s |
| 600M | **OOM** | 10,731 M rows/s |
| 1.2B | **OOM** | 13,358 M rows/s |
| 2.0B | **OOM** | 10,799 M rows/s |

Past one GPU's memory, single-GPU cuDF stops running at all. That's the boundary. It's a distribution win over cuDF's kernels, not a single-GPU compute win, and it's why Batcher uses cuDF as the per-GPU data plane rather than trying to out-code it.

## Reproduce

```bash
python benchmarks/run.py --benchmark operators --tier single --scale 1
python benchmarks/run.py --benchmark tpch      --tier single --scale 1
```

## See also

- {doc}`analytics` for the full operator table with DuckDB alongside.
- {doc}`vs-duckdb` and {doc}`vs-daft` for the other single-node comparisons.
- {doc}`../deep-dives/sort-internals` for the fused top-N heap behind the 50x, and the sample-sort that narrowed the float-sort loss.
- {doc}`../deep-dives/window-internals` for why `lag()` is ahead and the partitioned `SUM` is behind.
- {doc}`../deep-dives/gpu-execution` for cuDF as the per-GPU data plane.
- {doc}`../user-guide/sorting` for `top_k` and `sort` in the API.
- {doc}`methodology` for hardware and correctness gating.
