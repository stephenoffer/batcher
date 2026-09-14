# Does every single-node operation and optimization scale as nodes are added?

A code audit, 2026-09-11, of the question the title asks: Batcher is run single-node today,
so which of the things that make it fast single-node keep working when the work is spread
over N machines, and which are silently O(1) in N.

This is the companion to `distributed_scaling_audit.md`, which measured the **aggregate**
ladder on a real 4-worker cluster and found where it stops being linear. That document has
the numbers. This one has the coverage: every operator and every optimization, with the
scaling term named, so a gap is visible rather than discovered later.

**What is verified here is the code, not a cluster.** Nothing below is a new measurement.
Where `distributed_scaling_audit.md` has a measured figure it is cited; everything else is a
reading of the implementation, and the distinction is marked in every row.

A reading is the weaker evidence and this page should not pretend otherwise. The same pass
that produced it recorded a competitor as failing a suite on the strength of a seven-minute
cap, and a longer cap then got 26 of 27 cases through -- the figure had measured the cap.
A row below that says "yes" says the *shape* is right, and says nothing about the constant.

## What "scales linearly" has to mean

Three conditions, and the third is the one that actually bites:

1. **Per-node work is `O(rows / N)`.** Each node does its share and no more.
2. **Coordination is at worst `O(N)`**, and not on a critical path that also grows.
3. **No term is `O(rows)` on the driver.** A single node touching every row caps the
   cluster at that node, however many others there are.

An operator can satisfy 1 and 2 and still not scale, because a driver-side concatenation of
the result is `O(rows)`. That is the failure this audit looks for first.

## Operators

Every stateful operator is built as `partial -> combine -> finalize` (invariant #7), which
is what makes a distributed form possible at all. The table is what each one actually does.

| Operator | Distributed form | Driver term | Scales |
|---|---|---|---|
| `Scan` | Split-per-source, read on the worker holding the split | none | yes |
| `Filter`, `Project` | Row-wise, fused into the map prefix | none | yes |
| `Aggregate` | `partial_aggregate` -> hash shuffle -> `combine_finalize` on the reducers | result only | yes, measured |
| `Distinct` (whole row) | The aggregate shuffle verbatim: group by every column | result only | yes |
| `Distinct` (keyed) | Row shuffle, dedup per reducer | result only | yes |
| `HashJoin` (shuffle) | Both sides co-partitioned by key, per-bucket joins independent | result only | yes |
| `HashJoin` (broadcast) | Small side replicated, big side split, no shuffle | none | yes, but see below |
| `Sort` | Sample boundaries, range-partition, sort each range, concatenate in key order | **`O(rows)` unless `materialize=False`** | conditional |
| `Window` | Hash shuffle by partition key, whole partition on one reducer | result only | yes |
| `Union` | One shuffle when branches allow, else branch-by-branch | concat of branch results | yes |
| `Limit` / top-N | Per-node top-N, driver picks | `O(N x n)` | yes |
| `Unnest`, `Unpivot`, `Sample`, `RowId` | Row-wise pass-through under a breaker | none | yes |
| `AsofJoin`, `RangeJoin` | Share the join partitioner | result only | yes |

Two rows deserve their qualifiers.

**`Sort` was the largest driver term left.** A sort is row-preserving, so concatenating the
ranges on the driver costs the size of the whole relation on one node -- which is why
`iter_batches(distributed=True)` over an `ORDER BY` used to hold the entire result before
yielding a first batch. `materialize=False` returns a `MaterializedSource` over the range
buckets in leading-key order instead. The ordering survives because it was never produced by
the concatenation: each bucket is a range, globally ordered against every other, so reading
the files in that order *is* the row order. Callers that do not pass it still pay `O(rows)`.

**A broadcast join replicates the build side to every worker**, so its network cost is
`O(N x build)` and its per-node memory is `O(build)` regardless of N. That is the right
trade at small build sizes and it is not linear scaling -- it is constant per node, which is
better, until the build side grows. The planner's broadcast threshold is what keeps this
honest, and it is a cost-model decision rather than a structural guarantee.

## Optimizations

This is the half that is easy to get wrong, because an optimization that simply does not run
distributed costs nothing visible -- the query is merely slower than it looks single-node.

| Optimization | Distributed | Note |
|---|---|---|
| Morsel-driven parallelism | per node | `usable_cores` reads the actor's CPU affinity and cgroup quota, not `available_parallelism`, so a Ray worker sizes to what it actually has |
| Cranelift JIT | per node | compiled per operator on each worker |
| Chunked partials / partition choice | per node | `dist::partial_aggregate` makes the same choice the single-node executor does, on the worker's own share |
| Spill | per node | each worker spills its own partition |
| Learned stats (`MetadataHub`) | driver, consumed by all | workers report `ExecMetrics`; the hub learns once and every later plan benefits |
| Adaptive re-optimization | yes | `gating` turns `"auto"` on for a distributed shape the one-shot dispatcher would otherwise fix in advance |
| Plan cache | driver | plan-shaped, not data-shaped, so `O(1)` in N by construction |
| Predicate / projection pushdown | per node | applied to the plan before it is split, so every worker inherits it |
| **Common-subplan reuse** | **no** | **see below** |

### The one real gap: common-subplan reuse does not distribute

`api.subplan_reuse` computes a repeated subtree once and rewrites every appearance into a
scan of the result. It runs on the single-node relational executor only, and its own module
docstring says why: it materializes into an `InMemorySource` **on the driver**, where the
distributed route would have to keep the intermediate partitioned across the workers --
`staging`'s `MaterializedSource` machinery rather than this module's.

That makes it the clearest instance of the thing this audit is for. On TPC-DS single-node it
is worth 100 ms across the suite and turns q18 from 258 ms to 191. Distributed it is worth
nothing, and the queries it helps most -- `ROLLUP`, which repeats one input once per level --
are exactly the ones a cluster would be asked to run at scale. Closing it is not a new
mechanism: `staging.MaterializedSource` already keeps a partitioned intermediate across
workers for the adaptive loop, and this would splice the same thing from a structural
decision rather than a measured cardinality.

### Two known non-linear terms, already recorded

Both are from `distributed_scaling_audit.md` and are repeated here so this page is a complete
register rather than a partial one:

* **The reducer floor made the exchange `O(workers squared)` on a small aggregate.** Fixed.
* **The combiner tree builds an `O(reducers x sources)` structure on the driver.** Open.
  This is the term that grows with *both* halves of the cluster at once, and it is the one to
  watch as N rises.

## What this audit did not do

No cluster run. Every row above is a reading of the implementation except the aggregate
ladder, which is cited from the measured pass. A reading establishes that the shape is right;
it does not establish the constant, and three of the rows (`Sort` with `materialize=True`,
broadcast join at a large build side, the combiner tree) have constants that decide whether
the shape matters.

It also did not audit the **streaming** path, the **device tier**, or `graph`/`ml` surfaces.
Those compose the same primitives, but "composes the same primitives" is the claim this page
exists to stop taking on trust.
