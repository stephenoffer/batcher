# vs Polars

This page compares Batcher against Polars on single-node analytics.

The split is unusually sharp. Batcher takes sorting, top-N, filtered counts, and most of the window family by very large margins, up to 50x, and takes the TPC-H suite at sf10 by 2.26x.

:::{important}
Polars' TPC-H q6 returns the **wrong revenue**, folding the bound `0.06 + 0.01` to `0.06999999999999999` in IEEE double, which drops every `l_discount = 0.07` row, and the
harness declines to time a query whose result does not match the oracle. So Polars has no
number on q6 rather than a fast one. That gate runs before every timing on this page.
:::

## Scorecard

Each row is one workload shape, with the engine that won it and by how much. Read them
against the methodology above rather than in isolation:

| Shape | Winner |
|---|---|
| Sort → top-N (`LIMIT`) | Batcher, 50× |
| Window {py:func}`lag() <batcher.lag>` | Batcher, 17× |
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

## Hash and quantile paths

All on 16 cores, in-memory Arrow, correctness-gated against DuckDB *and* Polars. These are the
shapes that moved most, and the direction of travel is steep:

| Operator | Batcher now | Batcher before |
|---|---:|---:|
| group-by, high cardinality (5M rows -> 1.25M groups) | **182 ms** | 400 ms |
| `DISTINCT` (1.25M distinct ints, 5M rows) | **111 ms** | 300 ms |
| full sort `ORDER BY <f64>` (2M rows) | **68 ms** | 164 ms |
| two-key sort `ORDER BY <i64>, <i64>` (2M rows) | **91 ms** | 561 ms |

The high-cardinality group-by came down when the radix combine hashed native keys directly and
merged its partitions in parallel; the float sort came down on sample-sort; and the two-key
sort was single-threaded before. Measured against Polars, those four ratios improved from 4.7x
to 2.1x, 1.6x to 1.4x, 4.9x to 1.7x, and 8.9x to 1.4x.

## Join planning

Polars' benchmark suite writes its TPC-H queries as hand-ordered DataFrame pipelines. Batcher takes the SQL and chooses the join order itself, through a bushy dynamic-programming search over the join graph. On q5, a six-table join, that search is worth a large multiple.

Forcing Batcher to execute the join order Polars' own q5 is written in, rather than the one its optimizer picks:

| Join order on TPC-H q5 (sf1) | Batcher |
|---|---:|
| Chosen by Batcher's optimizer | **40.5 ms** |
| Polars' hand-written order | 591.1 ms |

Both return the same five rows and the same revenue. The optimizer's plan is **14.6x** faster than the hand-written one, which is the clearest available measure of what the join search contributes: on a six-table query, ordering decides the runtime, and getting it from a cost model beats getting it from an author.

Read this as a statement about join ordering, not about Polars. Those pipelines are `LazyFrame`s, so Polars is free to reorder them internally, and it runs q5 in 14.4 ms. The comparison above holds Batcher's engine fixed and varies only the order it is given, which is what isolates the planner's contribution from the kernels'.

## SQL

Polars can't parse most of the TPC-H suite through its SQL frontend, including multi-table `FROM`, `EXISTS`, and non-equi joins, so its column in the results is mostly `ERR`. That says something about its SQL surface and nothing about its speed, and it's reported that way.

Where it does parse q6, it computes the **wrong revenue**, folding the bound `0.06 + 0.01` to `0.06999999999999999` in IEEE double, which drops every `l_discount = 0.07` row.
A wrong answer gets no timing, so Polars gets no number there rather than a fast one.
Batcher matches DuckDB on all 22 queries.

## GPU

:::{note}
The table below was measured on an 8xT4 cluster. Every table above it was measured on a single 16-core node. Those are different machines running different work, so a row from one can't be set against a row from the other. See {doc}`/benchmarks/methodology`.
:::

Polars' GPU mode runs on cuDF, so the like-for-like comparison is against cuDF itself. On a group-by sum over 1000 groups on an 8xT4 cluster, single-GPU cuDF is genuinely fast. For data that fits one GPU it beats Batcher's distributed cuDF path, because the cross-device combine isn't free:

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

- {doc}`/benchmarks/results/analytics` for the full operator table with DuckDB alongside.
- {doc}`/benchmarks/comparisons/vs-duckdb` and {doc}`/benchmarks/comparisons/vs-daft` for the other single-node comparisons.
- {doc}`/architecture/deep-dives/operators/sort-internals` for the fused top-N heap behind the 50x, and the sample-sort behind the float-sort result.
- {doc}`/architecture/deep-dives/operators/window-internals` for why `lag()` is ahead.
- {doc}`/architecture/deep-dives/distribution/gpu-execution` for cuDF as the per-GPU data plane.
- {doc}`/user-guide/transform/rows/sorting` for `top_k` and `sort` in the API.
- {doc}`/benchmarks/methodology` for hardware and correctness gating.
