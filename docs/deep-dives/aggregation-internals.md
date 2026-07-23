# Aggregation internals

`GROUP BY` is the operator the engine is best at, and the one where the interesting decisions
are made at runtime rather than at plan time. This page follows a `group_by(...).agg(...)`
from the morsel to the output rows.

The algebra it obeys (`partial → combine → finalize`, with `combine` associative and
commutative) is covered in [Mergeable algebra](mergeable-algebra.md). This page covers what
those three functions do, and the one decision the executor refuses to take on faith.

## Step 1: assign each row a dense group id

`assign_groups` (`crates/bc-runtime/src/agg/group/assign.rs`) is the hot path of every hash
aggregate, `DISTINCT`, and partitioned window. It maps each row to a `u32` group id and
returns the group count and the distinct key columns in first-seen order.

Which strategy runs depends on the key, and the dispatch falls into three families:

| Family | Taken when | How the group id is found | Cost |
|---|---|---|---|
| Dense direct map | a non-nullable integer-like key whose value span fits the dense budget: dictionary codes, dense ids, enums, and the canonical bits of a non-null `Float64` | one linear pass for `(min, max)`, then `value - min` through a direct-indexed table | no hashing at all |
| Typed hash | a single `Int64`, `Utf8`/`Binary`, or non-null `Float64` key, and multi-column keys the engine can pack natively | hash the native values or bytes directly | one hash, no encode |
| Row encoding | everything else: nullable floats, and mixed multi-column keys the packers decline | Arrow's `RowConverter` into a comparable byte string, then hash | a per-row encode and an allocation |

The dense budget is `4 × rows`, clamped to between 1,024 and 2^20 slots. The upper cap keeps
the `u32` table under 4 MiB, and the lower clamp keeps the dense path reachable for a small
morsel. A non-null `Float64` key reaches these fast paths through the `-0.0` and NaN
canonicalization in `keys.rs`, so the fast path and the encoder agree on group identity.

The fast paths exist because that per-row encode is the difference between a group-by that
keeps up with DuckDB and one that doesn't.

## Step 2: accumulate (`partial`)

With dense group ids in hand, each aggregate scatter-adds into its own per-group state array.
The naive shape is one pass per aggregate, which streams the `group_ids` array N times for N
aggregates.

`crates/bc-runtime/src/agg/fused.rs` fuses the *simple scalar* aggregates (`sum`, `count`,
`count(*)`, `min`, `max`, `mean`) into a single linear scan that visits each row once and
updates every fused accumulator. It is a pure loop interchange of independent scatter-adds:
each accumulator owns only its own state, and the fused loop visits rows in the same
`0..num_rows` order as the per-call kernels, so the per-(group, column) sequence of operations
is unchanged and the result is element-for-element identical. The complex aggregates
(variance, median, `arg_min`/`arg_max`, covariance, the sketches) keep their own per-call
pass, as do two-input aggregates, which decline fusion outright.

`partial` emits **state**, not answers. `mean` emits `(sum, count)`. `median` emits the
group's values as a `List`. `approx_count_distinct` emits an HLL register array. When a partial
crosses the distributed boundary its state columns are given synthetic names of the form
`__s{aggregate_index}_{state_column_index}` (`crates/bc-interp/src/dist.rs`), so the partial
travels as an ordinary Arrow batch across a thread, a spill file, or a network hop.

## Step 3: combine

`combine` concatenates the partials' key columns and state columns, regroups by key, and
merges each aggregate's state with its associative reducer.

Above `radix_parallel_threshold` (default 200,000 concatenated rows, set on `RuntimeTuning`
in `bc-arrow` and mirrored on `EngineConfig` in `bc-ir`) it takes the parallel path,
`combine_radix` (`agg/group/combine.rs`): hash-radix partition the concatenated partials
by key so every row of a group lands in one partition, then group *and* merge each partition
independently across threads with **no cross-partition merge**. The otherwise-serial per-group
accumulate scan, which dominates a many-group combine, becomes a parallel one. Group *order*
differs from the serial path, which callers already treat as unspecified for a hash aggregate.

## The decision the executor refuses to guess

`partial → combine` is the right shape when grouping *reduces*. `GROUP BY l_returnflag` turns
a 16,384-row morsel into 3 rows and the merge is trivial.

It's the wrong shape when grouping doesn't reduce. `GROUP BY l_orderkey` over TPC-H
`lineitem` yields ~4 rows per group, so a morsel's partial has ~4,096 groups and the merge
inherits nearly the whole relation. `GROUP BY l_orderkey, l_linenumber` reduces *nothing*:
every partial row survives, and `combine` concatenates 60M rows of keys and states, hashes
them, bins them, and gathers them again. Measured at scale factor 10: the entire per-morsel
hash build (~60M inserts) is thrown away, and `combine` costs ~35 ns per *partial row*:
2.25 s for a group-by DuckDB answers in 396 ms.

So there is a second shape, `partition → partial → finalize`
(`crates/bc-interp/src/agg_par.rs`): hash-partition the input morsels by group key first, then
aggregate each partition exactly once. Equal keys co-locate, so the partitions are key-disjoint
and each one's partial is already final. `combine([p]) ≡ p`, and the union of the partitions
is the answer. One hash build over the relation instead of two, one gather instead of three.

```text
  A.  partial → combine            grouping REDUCES  (GROUP BY l_returnflag)

      m0 ──partial──► [3 rows] ┐
      m1 ──partial──► [3 rows] ├──► combine ──► finalize ──► 3 rows
      m2 ──partial──► [3 rows] ┘
      a hash build per morsel, and the merge is trivial


  B.  partition → partial          grouping does NOT reduce  (GROUP BY l_orderkey)

      m0 ┐                        ┌─► bucket 0 ──partial──► already final ─┐
      m1 ├── hash-partition ─────►├─► bucket 1 ──partial──► already final ─┼─► union
      m2 ┘   by the group key     └─► bucket 2 ──partial──► already final ─┘

      equal keys co-locate, so combine([p]) ≡ p and there is nothing to merge.
      one hash build over the relation instead of two; one gather instead of three.
```

This isn't a second aggregation semantics. It's the exact composition `bc-interp::dist` runs
across machines, executed across cores.

**How the choice is made: by measurement, not estimation.** The optimizer's `ndv` for a group
key is a sketch, and after a filter it is a guess about a distribution nobody has looked at.
So the executor partials a *sample* of morsels (work the reducing path needs anyway) and
reads the reduction those partials actually achieved. Below `REDUCTION_CEILING` (0.20 partial
rows kept per input row) the sample says grouping reduces, and its partials are handed straight
back to the standard path, unwasted.

The threshold is measured rather than argued. Aggregating a 60M-row table on 96 cores over a synthetic
key of varying cardinality (milliseconds, lower is better):

| rows kept per input row | 0.012 | 0.049 | 0.100 | 0.182 | 0.342 | 0.683 | 0.999 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `partial → combine` | 39.5 | 70.7 | 125.7 | 197.6 | 350.9 | 745.6 | 1351.6 |
| `partition → aggregate` | 274.8 | 225.7 | 198.2 | 187.6 | 185.3 | 189.7 | 370.2 |

They cross just under 0.18. Rounding to 0.20 keeps the reducing path wherever it is clearly
better and concedes only near-ties.

Memory has the last word. The partition path holds the gathered relation in memory where the
reducing path can spill its partials through grace partitioning, so `par` admits the partition
path's footprint against the memory pool first and keeps the bounded shape when the pool says
no. Under pressure, bounded beats fast.

## Spilling is the same algebra

`agg/spill.rs` bounds peak memory to one hash partition. Per-morsel partials are routed to one
of P partitions by a hash of the group key and written to a `SpillStore`; because a key always
hashes to the same partition, running `combine` + `finalize` one partition at a time produces
the global aggregate. `MemSpillStore` keeps the partitions in memory (used to prove the grace
algebra matches the oracle); `DiskSpillStore` streams them to Arrow IPC files, optionally
compressed. The IPC stream self-describes its compression, so the codec choice trades CPU for
bytes and cannot change a result.

## DISTINCT

`DISTINCT` is an aggregate with no aggregates: assign group ids over all columns, keep one row
per group. It gets one specialization worth knowing about. `distinct_dense`
(`agg/distinct.rs`) handles a whole-relation `DISTINCT` over a single dense integer column
without hashing, partitioning, or materializing the input. Which values occur is a presence
bitmap indexed by `value - min`, each morsel-chunk ORs into its own bitmap across cores, the
bitmaps reduce with a word-wise OR, and the distinct values fall out of a scan of the set bits.
Two linear passes over the key and nothing else. It declines, and the general path runs, unless
the input is exactly one `Int64` column with no nulls whose value span fits the dense budget.

:::{warning}
`DISTINCT` has no defined row order, and the three paths genuinely differ: ascending value
order from `distinct_dense`, first-seen order from the sequential oracle, bucket order from the
parallel dedup. Don't depend on one. If you need an order, sort.
:::

## What it looks like

Both front-ends lower to the same `Aggregate` node, so both take every decision on this page.

::::{tab-set}
:::{tab-item} DataFrame
```python
import batcher as bt

ds = bt.from_pydict(
    {
        "region": ["east", "west", "east", "west", "east"],
        "amount": [10.0, 20.0, 30.0, 40.0, 50.0],
        "units": [1, 2, 3, 4, 5],
    }
)

out = (
    ds.group_by("region")
    .agg(
        n=bt.count(),
        total=bt.col("amount").sum(),
        avg=bt.col("amount").mean(),          # state is (sum, count), not an average
        hi=bt.col("units").max(),
    )
    .sort("region")
    .to_pydict()
)
print(out)
```
:::

:::{tab-item} SQL
```python
print(
    bt.sql(
        """
        SELECT region,
               COUNT(*)      AS n,
               SUM(amount)   AS total,
               AVG(amount)   AS avg,
               MAX(units)    AS hi
        FROM t
        GROUP BY region
        ORDER BY region
        """,
        t=ds,
    ).to_pydict()
)
```
:::
::::

```text
{'region': ['east', 'west'], 'n': [3, 2], 'total': [90.0, 60.0], 'avg': [30.0, 30.0], 'hi': [5, 4]}
```

## Where it stands

On the operator benchmarks (16 cores, TPC-H `lineitem` at scale factor 1): group-by sum on one
key runs in 7.6 ms against DuckDB's 10.0 ms and Polars' 17.1 ms; two keys, 11.6 ms against 16.9
and 28.8. A global sum is 0.5 ms against 2.7 and 1.8. This is the shape the engine wins.

The sampling decision above pays for a sample of partials it may throw away; the ceiling is set
where that waste is cheaper than the wrong shape.

:::{tip}
The exact list-state aggregates (`median`, `count_distinct`) hold every value of every group,
so a high-cardinality median is memory-hungry by construction. When that bites, reach for
`approx_quantile` and `approx_count_distinct`: they carry a bounded-error sketch as their
state instead, and merge in constant space.
:::

## Where the code lives

- `crates/bc-runtime/src/agg/mod.rs`: `AggFunc`, `partial`, `combine`, `finalize`
- `crates/bc-runtime/src/agg/group/assign.rs`: dense ids, the three key paths
- `crates/bc-runtime/src/agg/group/combine.rs`: the parallel radix regroup
- `crates/bc-runtime/src/agg/fused.rs`: the fused scalar accumulators
- `crates/bc-runtime/src/agg/spill.rs`: grace aggregation
- `crates/bc-runtime/src/agg/{var,median,qsketch,hll,stats,argextreme,distinct}.rs`: the state shapes
- `crates/bc-interp/src/agg_par.rs`: the measured partition-vs-preaggregate decision

## See also

:::{seealso}
- [Architecture](../architecture/index.md): where an operator's state is allowed to live
- [Execution engine](../internals/execution.md): the operator the plan node lowers to
- `docs/internals/mathematical_foundations.md` (in the repo, not a site page): the sketch error bounds behind `approx_*`
- [Aggregations](../user-guide/aggregations.md): the API this page is under
- [Distinct and dedup](../user-guide/distinct-and-dedup.md): the `DISTINCT` surface
- [Analytics benchmarks](../benchmarks/analytics.md): the group-by numbers quoted above
- [Mergeable algebra](mergeable-algebra.md): why any of this is allowed to run in parallel
- [Morsel parallelism](morsel-parallelism.md): where the morsels come from
- [Spilling](spilling.md): what happens when the state does not fit
:::
