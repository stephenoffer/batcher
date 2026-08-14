# Aggregation internals

`GROUP BY` is the operator the engine is best at, and the one where the interesting decisions
are made at runtime rather than at plan time. This page follows a {py:meth}`group_by(...).agg(...) <batcher.Dataset.group_by>`
from the morsel to the output rows.

The algebra it obeys (`partial → combine → finalize`, with `combine` associative and
commutative) is covered in {doc}`Mergeable algebra </architecture/deep-dives/operators/mergeable-algebra>`. This page covers what
those three functions do, and the one decision the executor refuses to take on faith.

## Step 1: assign each row a dense group id

`assign_groups` (`crates/bc-runtime/src/agg/group/assign.rs`) is the hot path of every hash
aggregate, `DISTINCT`, and partitioned window. It maps each row to a `u32` group id and
returns the group count and the distinct key columns in first-seen order.

Which strategy runs depends on the key, and the dispatch falls into three families:

| Family | Taken when | How the group id is found | Cost |
|---|---|---|---|
| Sorted runs | tried first, on any key the engine can prove arrives in sorted order | compare each row with its predecessor; equal keys are adjacent, so a run is a group | no hashing and no table |
| Dense direct map | a non-nullable integer-like key whose value span fits the dense budget: dictionary codes, dense ids, enums, and the canonical bits of a non-null `Float64` | one linear pass for `(min, max)`, then `value - min` through a direct-indexed table | no hashing at all |
| Typed hash | a single `Int64`, `Utf8`/`Binary`, or non-null `Float64` key, and multi-column keys the engine can pack natively | hash the native values or bytes directly | one hash, no encode |
| Row encoding | everything else: nullable floats, and mixed multi-column keys the packers decline | Arrow's `RowConverter` into a comparable byte string, then hash | a per-row encode and an allocation |

The dense budget is `4 × rows`, clamped to between 1,024 and 2^20 slots. The upper cap keeps
the `u32` table under 4 MiB, and the lower clamp keeps the dense path reachable for a small
morsel. A non-null `Float64` key reaches these fast paths through the `-0.0` and NaN
canonicalization in `keys.rs`, so the fast path and the encoder agree on group identity.

The fast paths exist because that per-row encode is the difference between a group-by that
keeps up with DuckDB and one that doesn't.

### When the key arrives sorted

Sorted input makes equal keys adjacent, so a row's group is decided by comparing it with the
row before it. `crates/bc-runtime/src/agg/group/runs.rs` does that, and `assign_groups` tries
it before any hash path, because it replaces hashing rather than speeding it up.

The interesting part is where the engine gets the ordering from. It does not take anyone's word
for it. A sort key declared on a lakehouse table is metadata that nothing enforces on write, and
an aggregate that believes a false declaration does not return a slow answer, it returns a wrong
one: the same key, split across two non-adjacent runs, is emitted as two groups. So the engine
establishes the ordering itself, on every aggregate, whether or not anything declared it.

Establishing it is affordable because the check is chunked and stops at the first chunk holding
a violation. Unordered input is rejected after a few hundred rows, which is why this is
attempted unconditionally instead of being gated on a plan flag. A sampled check would be
cheaper still and is not safe: a key cycling `0,1,…,99,0,1,…` is ordered across any short
prefix, so sampling accepts it and then pays for a full detection pass that fails.

Either direction counts, since descending input clusters equal keys just as well as ascending.
A key containing nulls declines, because a null compares as null under `<`, which reads as
unordered.

Nothing about this changes the answer. Group ids come out in the same first-seen order the hash
paths produce, and float keys are canonicalized first so `-0.0` groups with `0.0` and every NaN
groups together, exactly as `GROUP BY` requires:

```python
import batcher as bt
from batcher import col

# A time-ordered ingest: the key arrives sorted, so the runs are the groups.
events = bt.from_pydict(
    {
        "day": [1, 1, 1, 2, 2, 3, 3, 3, 3],
        "amount": [5.0, 3.0, 2.0, 8.0, 1.0, 4.0, 4.0, 2.0, 6.0],
    }
)
print(events.group_by("day").agg(total=col("amount").sum()).sort("day").to_pydict())
```

```text
{'day': [1, 2, 3], 'total': [10.0, 9.0, 16.0]}
```

The same query over the same rows shuffled returns the same groups and the same totals; only
the path that found them differs.

Because `assign_groups` is shared, `DISTINCT` and the partitioned window get this at the same
time, without either operator knowing about it.

Worth stating plainly, because the mechanism measures far better than the query does: this is a
**small** win. One `partial` call over 6M sorted rows is up to 6x faster, but the engine
morselizes and never makes that call, so an A/B of two builds over the same data measures
1.0-1.2x end to end. What the design buys reliably is that it costs nothing when it does not
apply — unordered input measured at parity — which is what makes attempting it on every
aggregate the right default.

The state is still proportional to the group count. A sorted group-by can in principle hold one
group at a time, by emitting each group when its run closes, and that is not built: a group
emitted early cannot be withdrawn if a later batch reintroduces its key, so it is sound only
where the ordering is a fact about the plan rather than a claim about the data.

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
group's values as a `List`. {py:meth}`approx_count_distinct <batcher.plan.expr_ir.core.Expr.approx_count_distinct>` emits an HLL register array. When a partial
crosses the distributed boundary its state columns are given synthetic names of the form
`__s{aggregate_index}_{state_column_index}` (`crates/bc-interp/src/dist.rs`), so the partial
travels as an ordinary Arrow batch across a thread, a spill file, or a network hop.

## Step 3: combine

`combine` regroups the partials by key and merges each aggregate's state with its associative
reducer.

Below `radix_parallel_threshold` it takes the serial path: concatenate the partials' key and
state columns, then regroup the concatenation once. Above it, the parallel path `combine_radix`
(`agg/group/combine.rs`) hash-radix partitions by key so every row of a group lands in one
partition, then groups *and* merges each partition independently across threads with **no
cross-partition merge**. The otherwise-serial per-group accumulate scan, which dominates a
many-group combine, becomes a parallel one.

The parallel path never concatenates. Concatenating the partials only to hash and gather them is
a full copy of the key column that the merge never reads as one array, and on a high-cardinality
string key that copy is the merge's largest single cost. So `combine_radix` hashes each partial
in place, flattens the hashes in partial order, and gathers each partition's rows straight from
the partials through `(partial, row)` pairs. A companion, `combine_partitioned`, stops one step
earlier and hands the key-disjoint partitions back as separate morsels, so the executor's aggregate
tail emits those directly rather than gluing them into one batch and re-splitting for the next
operator.

The threshold is `radix_parallel_threshold`, set on `RuntimeTuning` in `bc-arrow` and mirrored on
`EngineConfig` in `bc-ir`. Its default, `0`, derives the crossover from the machine as
`partitions × 256`, because the parallel path's overhead is per *partition* (a bucket list, a gather, a hash
table) while the serial path's is per *row*, so the turn is a fixed number of rows per partition,
not one absolute count that is too high on a large box and too low on a small one. A positive
value pins it. Group *order* differs from the serial path either way, which callers already treat
as unspecified for a hash aggregate.

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

- {doc}`Architecture </architecture/index>`: where an operator's state is allowed to live.
- {doc}`Execution engine </architecture/internals/execution>`: the operator the plan node lowers to.
- `docs/architecture/internals/mathematical_foundations.md` (in the repo, not a site page): the sketch error bounds behind `approx_*`.
- {doc}`Aggregations </user-guide/analyze/aggregations>`: the API this page is under.
- {doc}`Distinct and dedup </user-guide/transform/rows/distinct-and-dedup>`: the `DISTINCT` surface.
- {doc}`Analytics benchmarks </benchmarks/results/analytics>`: the group-by numbers quoted above.
- {doc}`Mergeable algebra </architecture/deep-dives/operators/mergeable-algebra>`: why any of this is allowed to run in parallel.
- {doc}`Morsel parallelism </architecture/deep-dives/operators/morsel-parallelism>`: where the morsels come from.
- {doc}`Spilling </architecture/deep-dives/memory/spilling>`: what happens when the state does not fit.
