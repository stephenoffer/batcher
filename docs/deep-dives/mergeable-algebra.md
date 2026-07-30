# Mergeable algebra

*Mergeable algebra* is the rule that every stateful operator in Batcher is written once, as
three functions, and that the same three functions serve one core, many cores, bounded memory,
and many machines. This page describes those functions, the state shapes they force, and the
single definition of key identity they all depend on.

The failure it exists to prevent is specific. Write a stateful operator twice, once for a
single core and once for a cluster, and the two implementations eventually disagree on a float
key, or on nulls, or on ties. The bug only appears when the data is big enough to shuffle.

Every stateful operator in `bc-runtime` is built as three functions:

```text
partial(batch)   -> state
combine(states)  -> state         associative + commutative
finalize(state)  -> rows
```

Single-node execution is `finalize(partial(all_rows))`. Distributed execution is
`finalize(combine(partial(p) for each partition p))` after a shuffle by key. The *only*
difference is whether `combine` runs across partitions. There is no second distributed
operator, so there is no second set of semantics that could drift.

```text
   ONE CORE                MANY CORES                    MANY MACHINES
   ────────                ──────────                    ─────────────
   all rows                m0   m1   m2   m3             node A        node B
      │                     │    │    │    │              rows          rows
      │                     │    │    │    │                │             │
   partial              partial on each morsel           partial       partial
      │                     │    │    │    │                │             │
      │                     └──┬─┘    └─┬──┘         shuffle by hash(key)
      │                     combine   combine          ┌────┴────┐   ┌────┴────┐
      │                        └────┬────┘             │ combine │   │ combine │
      │                          combine               └────┬────┘   └────┬────┘
      │                             │                       │             │
   finalize                     finalize                finalize      finalize
      │                             │                       │             │
      ▼                             ▼                       ▼             ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                    the same rows. every time. by construction.          │
   └─────────────────────────────────────────────────────────────────────────┘
```

The invariant, stated as the test that must stay green:

```text
combine_finalize(partition(partial(p_k))) over all partitions  ==  single-node result
```

## Why associative *and* commutative

:::{important}
`combine` MUST be associative **and** commutative. Associativity lets partials merge in a tree
instead of a chain. Commutativity means the merge order doesn't matter, which is what makes
the result independent of thread scheduling and of network arrival order. If `combine` were
only associative, a cluster would have to impose a total order on its reducers' inputs, and
the answer would depend on which worker finished first.
:::

This is what forces the *shape* of the partial state. The state isn't the answer. It's
whatever is enough to compute the answer from any partition:

| Aggregate | Partial state | Finalize |
|---|---|---|
| `sum`, `min`, `max`, `count` | the value itself | identity |
| `mean` | `(sum, count)` | `sum / count` |
| `var`, `stddev` | Welford's `(mean, M2, count)` | Bessel-corrected variance |
| `median`, `quantile` | the group's non-null values as one `List` column | sort and index |
| `array_agg` | the group's non-null values as one `List` column | the list as-is, empty becomes null |
| `count_distinct` | the group's distinct values as one `List` column | union then count |
| `approx_count_distinct` | an HLL sketch | estimate |
| `approx_quantile` | a DDSketch | query |
| `corr`, `covar` | co-moments, as `(n, mean_x, mean_y, C2, M2x, M2y)` | the closed form |

`mean` emitting `(sum, count)` rather than an average is the whole idea in miniature: an
average of averages is wrong, a sum of sums over a sum of counts is right.

Variance carries Welford's `(mean, M2, count)` and merges it with Chan's parallel formula. The
obvious state, `(sum, sum_of_squares, count)`, is also mergeable, but it catastrophically
cancels when the mean is large relative to the spread. Mergeability alone isn't enough. The
state also has to stay numerically sound under merging.

`approx_quantile` carries a DDSketch rather than a KLL sketch for a reason that belongs on this
page: DDSketch's merge is **exactly** order-independent, so a distributed result is bit-identical
to a single-node one. KLL's compaction is order-sensitive and would agree only within its error
bounds, which breaks the guarantee this whole design exists to give. KLL still ships in
`bc-sketches`, where Kyber uses it for cardinality estimates, and an estimate that varies within
its bounds costs nothing.

The list-state aggregates (`median`, `count_distinct`) are **exact and mergeable, at the cost
of memory linear in the group's values**. That is a real trade. When you can't afford it,
`approx_count_distinct` and `approx_quantile` give you a bounded-error sketch state instead
(`crates/bc-sketches/`), which merges in constant space with a fixed seed so partition-built
sketches merge identically.

## One canonical key

Mergeability is worthless if two code paths disagree about what makes two keys "the same".
The group assigner, the radix combine, the shuffle, the join, and the window are separate code
paths for performance reasons, but they answer one semantic question, so the answer lives in
exactly one place: `crates/bc-runtime/src/keys.rs`.

:::{warning}
Getting this wrong doesn't reorder rows. It splits a group. If the shuffle disagrees with the
assigner about key identity, two rows that are one group land on different reducers and the
query returns **two groups where the oracle returns one**. Both directions have actually
happened here: a float key split across `-0.0` and `0.0`, because Arrow's `RowConverter`
encodes them to different bytes, and null integer keys scattered across every bucket.
:::

So `keys.rs` fixes the policy once:

```rust
// canonical u64 key bits for an f64: all NaNs are one group, +/-0.0 are one group
fn canon_f64(v: f64) -> u64 {
    if v.is_nan()      { 0x7ff8_0000_0000_0000 }   // one canonical quiet NaN
    else if v == 0.0   { 0 }                        // folds -0.0 into 0.0
    else               { v.to_bits() }
}

// one fixed hash for null keys, so every null row lands in one partition
const NULL_HASH: u64 = 0xa5a5_5a5a_dead_beef;
```

`canonicalize_float_keys` rewrites float key columns into canonical form *before* the general
shuffle path encodes them, so `RowConverter` and the raw-hash fast paths can't disagree. And
`float_total_cmp` gives `min`/`max` the same total order `ORDER BY` sorts in (NaN last),
because otherwise `max(x)` would silently ignore NaN and contradict
`SELECT x ORDER BY x DESC LIMIT 1` on the same column.

You can see the policy from the API:

```python
import batcher as bt

# -0.0 and 0.0 are one group; all NaNs are one group. This matches DuckDB.
d = bt.from_pydict({"k": [0.0, -0.0, float("nan"), float("nan")], "v": [1, 2, 3, 4]})
print(d.group_by("k").agg(n=bt.count()).to_pydict())
```

```text
{'k': [0.0, nan], 'n': [2, 2]}
```

## The same algebra, four ways

The point of doing this once is that the same three functions serve every execution mode.

::::{tab-set}
:::{tab-item} One core
`bc-interp::execute` calls `partial` on the whole input and `finalize`. The sequential oracle,
and the answer everything else is compared against.
:::

:::{tab-item} Many cores
`bc-interp::par` calls `partial` on each morsel in parallel, then `combine`, then `finalize`.
Same functions, different scheduler.
:::

:::{tab-item} Bounded memory
`agg::spill` (grace aggregation) routes per-morsel partials to one of P partitions by a hash
of the group key and writes them to a `SpillStore`. Because a key always hashes to the same
partition, every partial row for a group lands together, so running `combine` + `finalize`
**one partition at a time** is the global aggregate, with peak memory bounded to one
partition.

This isn't a special spilling algorithm. It's the distributive equivalence property, used
locally to bound memory. The same grace machinery, on the `PARTITION BY` keys, bounds a window
(`crates/bc-interp/src/window_spill.rs`).
:::

:::{tab-item} Many machines
`bc-interp::dist` exposes `partial_aggregate`, `partition_batches`, and `combine_finalize` at
the granularity a Ray orchestrator can map over partitions. The Python side in
`python/batcher/dist/` composes them. It is the same `bc-runtime` code underneath.
:::
::::

Disk and network are two sinks for one mechanism, and the parallel executor's in-memory
bucket shuffle is that mechanism with neither.

## Proving it, not asserting it

Three layers of test hold this up, and none of them are optional.

1. **Rust unit tests** in `bc-runtime` assert the mergeable invariant directly: partial each
   partition, combine in an arbitrary order, finalize, and compare against the single-node
   result.
2. **`seq == par`**. The parallel executor's output must equal the sequential oracle's, as a
   multiset for unordered relations and exactly for ordered ones.
3. **Differential vs DuckDB**, in `tests/differential/`. If Batcher and DuckDB disagree, Batcher
   is wrong until proven otherwise.

The cross-product matters more than any single case. The bugs that got through were not
"aggregation is broken". They were an operator with a non-default flag on a non-default
execution path: `sort(descending=True)` under spill, a distributed `GROUP BY` on a float key.
`tests/differential/test_diff_operator_matrix.py` exists to run
`{collect, spill, iter_batches, distributed}` x `{nulls, empty, one row, duplicates, -0.0/NaN,
descending}` for exactly this reason.

The user-visible consequence, which is the whole point:

```python
import dataclasses
import batcher as bt
from batcher import Config, config_context

ds = bt.from_pydict({"g": [i % 3 for i in range(10_000)], "x": list(range(10_000))})
base = Config()

def run(morsel_rows, parallelism):
    cfg = base.replace(
        execution=dataclasses.replace(
            base.execution, morsel_rows=morsel_rows, parallelism=parallelism
        )
    )
    with config_context(cfg):
        return ds.group_by("g").agg(s=bt.col("x").sum()).sort("g").to_pydict()

one_core = run(1024, 1)     # one partial, no combine
eight = run(256, 8)         # ~40 partials, combined in an arbitrary order
print(one_core == eight, one_core)
```

```text
True {'g': [0, 1, 2], 's': [16668333, 16661667, 16665000]}
```

## The rule when you add an operator

A stateful operator without a mergeable form caps the engine at a single node. That isn't an
acceptable trade here, and it's why the `add-relational-operator` and
`add-distributed-operator` skills both start at `bc-runtime`: write `partial`/`combine`/
`finalize`, prove `combine` associates and commutes, and the parallel path, the spill path,
and the distributed path all follow from it.

If your operator genuinely has no mergeable form, that's a design conversation, not a `TODO`.

## Where the code lives

- `crates/bc-runtime/src/agg/mod.rs`: `partial`, `combine`, `finalize`, `AggFunc`
- `crates/bc-runtime/src/keys.rs`: the one canonical key policy
- `crates/bc-runtime/src/agg/spill.rs`: grace aggregation (the same algebra, bounded)
- `crates/bc-interp/src/dist.rs`: the distributed primitives
- `crates/bc-sketches/`: mergeable HLL / KLL / Count-Min, fixed seed

## See also

:::{seealso}
- {doc}`Architecture <../architecture/index>`: the invariant this page is the implementation of
- `docs/internals/mathematical_foundations.md` (in the repo, not a site page): the algebraic statement and its proofs
- {doc}`Execution engine <../internals/execution>`: where `partial`/`combine`/`finalize` are called from
- {doc}`Aggregations <../user-guide/aggregations>`: the surface this algebra is hiding behind
- {doc}`Scaling benchmarks <../benchmarks/scaling>`: what bounded per-node memory buys as the cluster grows
- {doc}`Aggregation internals <aggregation-internals>`: how `partial` and `combine` actually run
- {doc}`Morsel parallelism <morsel-parallelism>`: the scheduling this algebra makes safe
- {doc}`Distributed scheduling <distributed-scheduling>`: the same three functions, mapped over Ray
:::
