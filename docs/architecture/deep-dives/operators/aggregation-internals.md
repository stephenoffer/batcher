# Aggregation internals

`GROUP BY` is the operator the engine is best at, and the one where the interesting decisions are made at runtime rather than at plan time. This page follows a {py:meth}`group_by(...).agg(...) <batcher.Dataset.group_by>` from the morsel to the output rows. The algebra it obeys, `partial → combine → finalize`, is covered in {doc}`Mergeable algebra </architecture/deep-dives/operators/mergeable-algebra>`.

Both front-ends lower to the same `Aggregate` node, so both take every decision on this page:

::::{tab-set}
:::{tab-item} DataFrame
```python
import batcher as bt
from batcher import col

ds = bt.from_pydict(
    {
        "region": ["east", "west", "east", "west", "east"],
        "amount": [10.0, 20.0, 30.0, 40.0, 50.0],
        "units": [1, 2, 3, 4, 5],
    }
)
out = ds.group_by("region").agg(
    n=bt.count(),
    total=col("amount").sum(),
    avg=col("amount").mean(),  # state is (sum, count), not an average
    hi=col("units").max(),
)
print(out.sort("region").to_pydict())
# {'region': ['east', 'west'], 'n': [3, 2], 'total': [90.0, 60.0], 'avg': [30.0, 30.0], 'hi': [5, 4]}
```
:::

:::{tab-item} SQL
```python
print(
    bt.sql(
        "SELECT region, COUNT(*) AS n, SUM(amount) AS total, AVG(amount) AS avg, "
        "MAX(units) AS hi FROM t GROUP BY region ORDER BY region",
        t=ds,
    ).to_pydict()
)
# {'region': ['east', 'west'], 'n': [3, 2], 'total': [90.0, 60.0], 'avg': [30.0, 30.0], 'hi': [5, 4]}
```
:::
::::

## Step 1: assign each row a dense group id

`assign_groups` ([`crates/bc-runtime/src/agg/group/assign.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-runtime/src/agg/group/assign.rs)) is the hot path of every hash aggregate, `DISTINCT`, and partitioned window. It maps each row to a `u32` group id and returns the group count and the distinct keys in first-seen order. The strategy depends on the key, tried in this order:

| Family | Taken when | How the group id is found |
|---|---|---|
| Sorted runs | the key provably arrives in sorted order | compare each row with its predecessor; a run is a group, so no hashing and no table |
| Dense direct map | a non-nullable integer-like key whose value span fits the dense budget (dictionary codes, dense ids, enums, canonical non-null `Float64` bits) | `value - min` through a direct-indexed table, no hashing |
| Typed hash | a single integer, `Utf8`/`Binary` or non-null `Float64` key, and null-free multi-column integer and byte keys the engine can pack or rank | hash the native values or bytes directly |
| Row encoding | everything else | Arrow's `RowConverter` into a comparable byte string, then hash |

The dense budget is `4 × rows`, clamped to between 1,024 and 2^20 slots, which keeps the `u32` table under 4 MiB. The path is picked by the data, not the query. These two aggregates are spelled the same way and reach different implementations, because the first key's ranges multiply to something small enough to index directly and the second's don't:

```python
sales = bt.from_pydict(
    {
        "region": [1, 1, 2, 2, 1, 2],
        "channel": [10, 20, 10, 20, 10, 10],
        "sku": [910_003, 910_003, 720_017, 910_003, 720_017, 910_003],
        "amount": [5.0, 3.0, 2.0, 8.0, 1.0, 4.0],
    }
)
dense = sales.group_by("region", "channel").agg(total=col("amount").sum())
print(dense.sort("region", "channel").to_pydict())  # direct map, no hashing
# {'region': [1, 1, 2, 2], 'channel': [10, 20, 10, 20], 'total': [6.0, 3.0, 6.0, 8.0]}
hashed = sales.group_by("region", "channel", "sku").agg(total=col("amount").sum())
print(hashed.count())  # sparse third key: hashed, one pass per column
# 6
```

:::{dropdown} How a key of several columns is hashed
Batcher hashes column-major, as DuckDB and Polars do: seed one hash per row, then make one pass per column, folding that column's values into the running hashes. Each pass streams two contiguous slices with no per-row walk over a column list.

The hashes stay materialized for the whole assignment because a table resize rehashes every entry, and an entry holds a representative row index. Without the hash array, rehashing means reading that row back out of every key column. Storing the hash inside the entry instead quadruples the entry, and past a few hundred thousand groups the wider table costs more cache than it saves.

The table starts small, since an analytical `GROUP BY` is usually low-cardinality. Once the initial capacity fills, the observed groups-per-row density projects the final count and one reserve replaces the doubling cascade. Together these are worth 1.4x to 6.2x on composite integer keys with many groups, and 1.07x to 1.12x on the low-cardinality shape (`docs/architecture/internals/competitor_technique_review.md`).
:::

### When the key arrives sorted

Sorted input makes equal keys adjacent, so [`runs.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-runtime/src/agg/group/runs.rs) decides each row's group by comparing it with the row before. The engine establishes the ordering itself on every aggregate rather than trusting a declared sort key, because a false declaration would split one key into two groups. The check is chunked and stops at the first violation, so unordered input is rejected after a few hundred rows and costs nothing measurable. Either direction counts, a key containing nulls declines, and float keys are canonicalized first. `DISTINCT` and the partitioned window share `assign_groups`, so they get this too.

```python
events = bt.from_pydict(
    {"day": [1, 1, 1, 2, 2, 3, 3, 3, 3], "amount": [5.0, 3.0, 2.0, 8.0, 1.0, 4.0, 4.0, 2.0, 6.0]}
)
print(events.group_by("day").agg(total=col("amount").sum()).sort("day").to_pydict())
# {'day': [1, 2, 3], 'total': [10.0, 9.0, 16.0]}
```

## Step 2: accumulate (`partial`)

With dense group ids in hand, each aggregate scatter-adds into its own per-group state array. [`fused.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-runtime/src/agg/fused.rs) fuses the simple scalar aggregates (`sum`, `count`, `count(*)`, `min`, `max`, `mean`) into one linear scan that updates every accumulator per row. It's a pure loop interchange, so the result is element-for-element identical. Variance, median, `arg_min`/`arg_max`, covariance, the sketches, and two-input aggregates keep their own pass.

`partial` emits state, not answers: `mean` emits `(sum, count)`, `median` emits the group's values as a `List`, {py:meth}`approx_count_distinct <batcher.plan.expr_ir.core.Expr.approx_count_distinct>` emits an HLL register array. Across the distributed boundary the state columns are named `__s{aggregate_index}_{state_column_index}` ([`dist.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-interp/src/dist.rs)), so a partial travels as an ordinary Arrow batch over a thread, a spill file, or a network hop.

## Step 3: combine

`combine` regroups the partials by key and merges each aggregate's state with its associative reducer. Below `radix_parallel_threshold` it concatenates the partials and regroups once. Above it, `combine_radix` (`agg/group/combine.rs`) hash-radix partitions by key so every row of a group lands in one partition, then groups and merges each partition independently across threads with no cross-partition merge.

:::{dropdown} Details of the parallel combine
The parallel path never concatenates. On a high-cardinality string key that copy is the merge's largest single cost, so `combine_radix` hashes each partial in place and gathers each partition's rows straight from the partials through `(partial, row)` pairs. `combine_partitioned` hands the key-disjoint partitions back as separate morsels, so the next operator doesn't re-split one big batch.

`radix_parallel_threshold` lives on `RuntimeTuning` in `bc-arrow`, mirrored on `EngineConfig`. Its default, `0`, derives the crossover as `partitions × 256` (`MIN_ROWS_PER_RADIX_PARTITION`), because the parallel path's overhead is per partition while the serial path's is per row. A positive value pins it. Group order differs between the two paths, which is unspecified for a hash aggregate anyway.
:::

## Partition first when grouping doesn't reduce

`partial → combine` is right when grouping reduces: `GROUP BY l_returnflag` turns a 16,384-row morsel into 3 rows. When grouping doesn't reduce, as with `GROUP BY l_orderkey, l_linenumber`, every partial row survives and `combine` re-hashes the whole relation. So there is a second shape, `partition → partial → finalize` ([`agg_par.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-interp/src/agg_par.rs)): hash-partition the input by group key first, then aggregate each key-disjoint partition exactly once. It's the same composition `bc-interp::dist` runs across machines, executed across cores.

```text
  A.  partial → combine            grouping REDUCES  (GROUP BY l_returnflag)
      m0 ──partial──► [3 rows] ┐
      m1 ──partial──► [3 rows] ├──► combine ──► finalize ──► 3 rows
      m2 ──partial──► [3 rows] ┘

  B.  partition → partial          grouping does NOT reduce  (GROUP BY l_orderkey)
      m0 ┐                        ┌─► bucket 0 ──partial──► already final ─┐
      m1 ├── hash-partition ─────►├─► bucket 1 ──partial──► already final ─┼─► union
      m2 ┘   by the group key     └─► bucket 2 ──partial──► already final ─┘
```

The executor chooses by measurement, not estimation. It partials a sample of morsels, work the reducing path needs anyway, and reads the reduction those partials achieved. Below `REDUCTION_CEILING` (0.20 partial rows kept per input row) the sample's partials go straight back to the standard path. The partition path holds the gathered relation in memory, so it's admitted against the memory pool first, and the spillable shape wins when the pool says no.

:::{dropdown} The measured crossover
Aggregating a 60M-row table on 96 cores over a synthetic key of varying cardinality (milliseconds, lower is better):

| rows kept per input row | 0.012 | 0.049 | 0.100 | 0.182 | 0.342 | 0.683 | 0.999 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `partial → combine` | 39.5 | 70.7 | 125.7 | 197.6 | 350.9 | 745.6 | 1351.6 |
| `partition → aggregate` | 274.8 | 225.7 | 198.2 | 187.6 | 185.3 | 189.7 | 370.2 |

They cross just under 0.18, and 0.20 keeps the reducing path wherever it is clearly better.
:::

## What the state costs

Every aggregate's state is in one of four families, and the family decides how it behaves when one group is much larger than the rest:

| Family | State per group | Examples |
|---|---|---|
| Fixed accumulator | constant | `sum`, `count`, `min`, `max`, `mean`, `var`, `arg_min` |
| Sketch | bounded by a configured error | `approx_count_distinct`, `approx_quantile` |
| Counted values | one entry per distinct value | `mode`, `top_k` |
| Value list | one entry per row | `median`, `quantile`, `count_distinct`, `array_agg` |

{py:meth}`mode <batcher.plan.expr_ir.core.Expr.mode>` and {py:meth}`top_k <batcher.plan.expr_ir.core.Expr.top_k>` only ask how often a value occurs, so they carry each group's distinct values with their counts ([`counted.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-runtime/src/agg/counted.rs)). They are exact, counts add under `combine`, and ties resolve to the smaller value so the answer doesn't depend on partitioning:

```python
visits = bt.from_pydict(
    {
        "user": ["ana"] * 4 + ["bo"] * 3 + ["cy"] * 2,
        "page": ["home", "home", "docs", "home", "pricing", "docs", "pricing", "pricing", "docs"],
    }
)
top = visits.group_by("user").agg(fav=col("page").mode(), top_two=col("page").mode_top_k(2))
print(top.sort("user").to_pydict())
# {'user': ['ana', 'bo', 'cy'], 'fav': ['home', 'pricing', 'docs'],
#  'top_two': [['home', 'docs'], ['pricing', 'docs'], ['docs', 'pricing']]}
```

:::{tip}
The exact list-state aggregates (`median`, `count_distinct`) hold every value of every group. For a high-cardinality median, reach for `approx_quantile` and `approx_count_distinct`, which carry a bounded-error sketch and merge in constant space.
:::

## Spilling is the same algebra

`agg/spill/mod.rs` bounds peak memory to one hash partition. Per-morsel partials are routed to one of P partitions by a hash of the group key and written to a `SpillStore`. A key always hashes to the same partition, so `combine` + `finalize` one partition at a time is the global aggregate. `DiskSpillStore` streams partitions to Arrow IPC files, optionally compressed, and the codec can't change a result.

![What an aggregation holds, and what it spills. partial assigns dense group ids, through a hash table, a direct map, or no table at all when the key arrives sorted, then scatters one row of state columns per group; that state is not the answer, because mean carries a sum and a count, var carries mean, M2 and count, and median carries the group's values as a list, and only finalize turns one into a number. The spill phase routes every partial to one of P partitions by a hash of the group key and writes them as Arrow IPC, so nothing is kept resident and this is not an eviction policy, and because a key always hashes to the same partition a group is never split across two of them. The merge then reads one partition at a time, combines the states by key and finalizes, re-partitioning with a fresh salt when a partition is still over budget, so peak memory is one partition rather than one hash table. This is the same algebra the distributed path runs, with combine reading from disk instead of from the network.](/_static/diagrams/agg_spill_states.svg)

## DISTINCT

`DISTINCT` is an aggregate with no aggregates: assign group ids over all columns and keep one row per group. `distinct_dense` (`agg/distinct.rs`) specializes a whole-relation `DISTINCT` over one non-null `Int64` column whose span fits the dense budget: each core ORs values into a presence bitmap indexed by `value - min`, and the distinct values fall out of a scan of the set bits. `DISTINCT` has no defined row order, so sort when you need one:

```python
print(bt.from_pydict({"k": [3, 1, 3, 2, 1]}).distinct().sort("k").to_pydict())
# {'k': [1, 2, 3]}
```

## Where it stands

On the operator benchmarks (16 cores, TPC-H `lineitem` at scale factor 1), group-by sum on one key runs in 7.6 ms against DuckDB's 10.0 ms and Polars' 17.1 ms; on two keys, 11.6 ms against 16.9 and 28.8. A global sum is 0.5 ms against 2.7 and 1.8.

## Where the code lives

- [`crates/bc-runtime/src/agg/mod.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-runtime/src/agg/mod.rs): `AggFunc`, `partial`, `combine`
- [`crates/bc-runtime/src/agg/dispatch.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-runtime/src/agg/dispatch.rs): `accumulate` and `finalize`, the per-function tables
- [`crates/bc-runtime/src/agg/group/assign.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-runtime/src/agg/group/assign.rs): dense ids and the key dispatch
- [`crates/bc-runtime/src/agg/group/runs.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-runtime/src/agg/group/runs.rs): the sorted-run short-circuit
- [`crates/bc-runtime/src/agg/group/combine.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-runtime/src/agg/group/combine.rs): the parallel radix regroup
- [`crates/bc-runtime/src/agg/fused.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-runtime/src/agg/fused.rs): the fused scalar accumulators
- [`crates/bc-runtime/src/agg/spill/mod.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-runtime/src/agg/spill/mod.rs): grace aggregation
- `crates/bc-runtime/src/agg/{var,median,sketch,stats,argextreme,distinct,counted}.rs`: the state shapes
- [`crates/bc-interp/src/agg_par.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-interp/src/agg_par.rs): the measured partition-vs-preaggregate decision

## See also

- {doc}`Architecture </architecture/index>`: where an operator's state is allowed to live.
- {doc}`Execution engine </architecture/internals/execution>`: the operator the plan node lowers to.
- `docs/architecture/internals/mathematical_foundations.md` (in the repo, not a site page). It is the v1-era design paper with an errata list at its top, and where it and the code differ the code decides. It covers the sketch error bounds behind `approx_*`.
- {doc}`Aggregations </user-guide/analyze/aggregations>`: the API this page is under.
- {doc}`Distinct and dedup </user-guide/transform/rows/distinct-and-dedup>`: the `DISTINCT` surface.
- {doc}`Analytics benchmarks </benchmarks/results/analytics>`: the group-by numbers quoted above.
- {doc}`Mergeable algebra </architecture/deep-dives/operators/mergeable-algebra>`: why any of this is allowed to run in parallel.
- {doc}`Morsel parallelism </architecture/deep-dives/operators/morsel-parallelism>`: where the morsels come from.
- {doc}`Spilling </architecture/deep-dives/memory/spilling>`: what happens when the state does not fit.
