# Mergeable algebra

*Mergeable algebra* is the rule that a stateful operator is written once, and that the one implementation serves one core, many cores, bounded memory, and many machines. This page covers the three functions an aggregate is built from, the state shapes they force, how the other stateful operators reach the same guarantee, and the single definition of key identity they all share.

Aggregation is written as three functions:

```text
partial(batch)   -> state
combine(states)  -> state         associative + commutative
finalize(state)  -> rows
```

Single-node execution is `finalize(partial(all_rows))`. Distributed execution is `finalize(combine(partial(p) for each partition p))` after a shuffle by key. The only difference is whether `combine` runs across partitions, so there is no second distributed operator whose semantics could drift.

From the API, that means the schedule never shows up in the answer. One core with one partial and eight cores with dozens of partials merged in arbitrary order return the same rows:

```python
import dataclasses

import batcher as bt
from batcher import Config, config_context

ds = bt.from_pydict({"g": [i % 3 for i in range(10_000)], "x": list(range(10_000))})


def run(morsel_rows, parallelism):
    ex = dataclasses.replace(Config().execution, morsel_rows=morsel_rows, parallelism=parallelism)
    with config_context(Config().replace(execution=ex)):
        return ds.group_by("g").agg(s=bt.col("x").sum()).sort("g").to_pydict()


print(run(1024, 1) == run(256, 8), run(1024, 1))
# True {'g': [0, 1, 2], 's': [16668333, 16661667, 16665000]}
```

```text
   ONE CORE                MANY CORES                    MANY MACHINES
   ────────                ──────────                    ─────────────
   all rows                m0   m1   m2   m3             node A        node B
      │                     │    │    │    │                │             │
   partial              partial on each morsel           partial       partial
      │                     └──┬─┘    └─┬──┘         shuffle by hash(key)
      │                     combine   combine          ┌────┴────┐   ┌────┴────┐
      │                        └────┬────┘             │ combine │   │ combine │
      │                          combine               └────┬────┘   └────┬────┘
   finalize                     finalize                finalize      finalize
      ▼                             ▼                       ▼             ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                    the same rows. every time. by construction.          │
   └─────────────────────────────────────────────────────────────────────────┘
```

The invariant, stated as the test that must stay green:

```text
combine_finalize(partition(partial(p_k))) over all partitions  ==  single-node result
```

## Fold or partition

`DISTINCT` is the same shape with an empty aggregate list. A join, a partitioned window, and a sort hold state that doesn't fold (a hash table of build rows, a partition's row set, an ordering), so they *partition* instead, so that no fold is needed:

- A join co-partitions both sides by the join key, so bucket `i` of the left joins only bucket `i` of the right.
- A window partitions on its `PARTITION BY` keys, so a partition is computed wherever it lands.
- A sort range-partitions the leading key, so the buckets concatenated in range order are the sorted relation.

`bc-interp::dist` exposes both families: `partial_aggregate`/`combine_finalize` for the first, and `partition_batches`, `range_partition_batches` and `salted_partition_batches` for the second. Top-N is bounded by a shared cut-off (`bc-runtime/src/topn.rs`) instead.

Every one of them guarantees the same rows, column names, and column types as single-node. Four things the query itself doesn't pin down may differ: a float reduction reassociates, a window function may break a tie its `ORDER BY` leaves open differently, a `LIMIT` over an unordered relation may keep a different set of rows, and an `array_agg` with no `order_by` may list a group's elements in a different order (never a different multiset). Anything else that differs is a bug.

## Why associative *and* commutative

:::{important}
`combine` MUST be associative **and** commutative. Associativity lets partials merge in a tree instead of a chain. Commutativity makes the result independent of thread scheduling and network arrival order, so the answer never depends on which worker finished first.
:::

That forces the shape of the partial state. The state isn't the answer. It's whatever is enough to compute the answer from any partition:

| Aggregate | Partial state | Finalize |
|---|---|---|
| `sum`, `min`, `max`, `count` | the value itself | identity |
| `mean` | `(sum, count)` | `sum / count` |
| `var`, `stddev` | Welford's `(mean, M2, count)` | Bessel-corrected variance |
| `median`, `quantile` | the group's non-null values as one `List` column | sort and index |
| `array_agg` | the group's non-null values as one `List` column | the list as-is, empty becomes null |
| {py:meth}`count_distinct <batcher.plan.expr_ir.core.Expr.count_distinct>` | the group's distinct values as one `List` column | union then count |
| {py:meth}`approx_count_distinct <batcher.plan.expr_ir.core.Expr.approx_count_distinct>` | an HLL sketch | estimate |
| `approx_quantile` | a DDSketch | query |
| `corr`, `covar` | co-moments, as `(n, mean_x, mean_y, C2, M2x, M2y)` | the closed form |

`mean` emitting `(sum, count)` is the whole idea in miniature: an average of averages is wrong, while a sum of sums over a sum of counts is right.

The list-state aggregates are exact and mergeable at the cost of memory linear in the group's values. When that's too much, the sketch states have a size that doesn't grow with the row count:

```python
d = bt.from_pydict({"g": ["a"] * 100 + ["b"] * 100, "v": list(range(200))})
out = d.group_by("g").agg(
    exact=bt.col("v").count_distinct(),  # list state, exact
    approx=bt.col("v").approx_count_distinct(),  # fixed-size HLL state
)
print(out.sort("g").to_pydict())
# {'g': ['a', 'b'], 'exact': [100, 100], 'approx': [100, 100]}
```

:::{dropdown} Numerical soundness and merge-order independence
Variance carries Welford's `(mean, M2, count)` and merges with Chan's parallel formula. The obvious state, `(sum, sum_of_squares, count)`, is also mergeable but catastrophically cancels when the mean is large relative to the spread, so mergeability alone isn't enough.

`approx_quantile` carries a DDSketch rather than a KLL sketch because DDSketch's merge is exactly order-independent, so a distributed result is bit-identical to a single-node one. An HLL folds register-wise by `max` and a DDSketch sums counts in fixed logarithmic buckets, so both reach the same state in any merge order (`agg/mod.rs::approx_quantile_is_merge_order_independent`). KLL and TDigest compact and re-cluster as they merge, so they stay on the estimate side in [`crates/bc-sketches/`](https://github.com/stephenoffer/batcher/tree/main/crates/bc-sketches). A DDSketch stores only occupied logarithmic buckets: at the default 1% accuracy, values spanning `1e-9` to `1e9` occupy at most 2,074 buckets per sign.

A float sum commutes but doesn't associate, which is the reassociation exception above. The list state of `array_agg` combines by concatenation, which commutes only as a multiset, so an unordered `array_agg` promises the multiset and leaves the order open. `array_agg(order_by=...)` sorts at finalize, so its result doesn't depend on merge order. `median`, `quantile` and `count_distinct` sort or deduplicate first, so their answers are exact in any order.
:::

## One canonical key

Mergeability is worthless if two code paths disagree about what makes two keys the same. The group assigner, the radix combine, the shuffle, the join, and the window are separate code paths for performance, but they answer one semantic question, so the answer lives in one place: [`crates/bc-runtime/src/keys.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-runtime/src/keys.rs). If the shuffle disagreed with the assigner, one group would land on two reducers and the query would return two groups where the oracle returns one.

The policy is visible from the API: `-0.0` and `0.0` are one group, and all NaNs are one group, matching DuckDB.

```python
d = bt.from_pydict({"k": [0.0, -0.0, float("nan"), float("nan")], "v": [1, 2, 3, 4]})
print(d.group_by("k").agg(n=bt.count()).to_pydict())
# {'k': [0.0, nan], 'n': [2, 2]}
```

:::{dropdown} The canonical key, in Rust
The float half lives in `bc-arrow` (`float_ident.rs`), the lowest crate that both `bc-runtime` and `bc-expr` see, so grouping keys and scalar comparisons can't drift apart. `keys.rs` re-exports it beside the null rule:

```rust
// bc-arrow: canonical u64 key bits for an f64 (all NaNs one group, +/-0.0 one group)
pub fn canon_f64_bits(v: f64) -> u64 {
    if v.is_nan()      { CANONICAL_NAN_BITS_F64 }   // 0x7ff8_0000_0000_0000
    else if v == 0.0   { 0 }                        // folds -0.0 into 0.0
    else               { v.to_bits() }
}

// bc-runtime keys.rs: one fixed hash for null keys, so every null row lands in one partition
pub(crate) const NULL_HASH: u64 = 0xa5a5_5a5a_dead_beef;
```

`canonicalize_float_keys` rewrites float key columns into canonical form before the general shuffle path encodes them, so Arrow's `RowConverter` and the raw-hash fast paths agree. `float_total_cmp` gives `min`/`max` the same total order `ORDER BY` sorts in (NaN last), so `max(x)` agrees with `SELECT x ORDER BY x DESC LIMIT 1`.
:::

## The same algebra, four ways

The same three functions serve every execution mode. The figure puts the four on one bus, so what separates them shows up as the one thing it is: what carries a partial state from `partial` to `combine`.

![One operator, written once, and the four transports that carry its partial state. Along the top rail, partial(batch) takes rows in and returns a state that is not the answer, combine(states) merges those states associatively and commutatively into one merged state per group, and finalize(state) takes the state in and returns rows. Below it, four execution modes feed that same combine and differ in nothing but what carries the partial to it: one core (bc-interp::execute) carries nothing and has one partial with nothing to merge, many cores (bc-interp::par) carry a thread hand-off of one partial per morsel, bounded memory (agg::spill) carries an IPC spill file read one partition at a time, and many machines (bc-interp::dist) carry a Flight stream hash-partitioned by key. The test that must stay green is that combine_finalize(partition(partial(p_k))) equals the single-node result, because arrival order cannot change the answer.](/_static/diagrams/mergeable_algebra.svg)

::::{tab-set}
:::{tab-item} One core
`bc-interp::execute` calls `partial` on the whole input and `finalize`. This is the sequential oracle everything else is compared against.
:::

:::{tab-item} Many cores
`bc-interp::par` calls `partial` on each morsel in parallel, then `combine`, then `finalize`. Same functions, different scheduler.
:::

:::{tab-item} Bounded memory
`agg::spill` (grace aggregation) routes per-morsel partials to one of P partitions by a hash of the group key and writes them to a `SpillStore`. A key always hashes to the same partition, so running `combine` + `finalize` one partition at a time is the global aggregate, with peak memory bounded to one partition. The same grace machinery, on the `PARTITION BY` keys, bounds a window ([`crates/bc-interp/src/window_spill.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-interp/src/window_spill.rs)).

```python
from batcher.config import MemoryConfig

big = bt.from_pydict({"k": [i % 50 for i in range(2000)], "v": list(range(2000))})
q = big.group_by("k").agg(total=bt.col("v").sum()).sort("k")
with config_context(Config().replace(memory=MemoryConfig(max_memory_bytes=1))):
    spilled = q.to_pydict()  # a one-byte budget forces the out-of-core path
print(spilled == q.to_pydict())
# True
```
:::

:::{tab-item} Many machines
`bc-interp::dist` exposes `partial_aggregate`, `partition_batches`, and `combine_finalize` at the granularity a Ray orchestrator can map over partitions. [`python/batcher/dist/`](https://github.com/stephenoffer/batcher/tree/main/python/batcher/dist) composes them over the same `bc-runtime` code.
:::
::::

## Practical limits

- A partition always holds all of one group's partial state, so spilling is bounded by the largest group. For a fixed-size state such as `sum` that's one row; for a list state such as `median` or `array_agg` it's every value of the group. {doc}`Spilling </architecture/deep-dives/memory/spilling>` covers what happens when a bucket stays over budget.
- An operator with neither a mergeable form nor a partitioning would cap the engine at one node, which is why the `add-relational-operator` and `add-distributed-operator` skills both start at `bc-runtime`.

## How it's tested

1. **Rust unit tests** in `bc-runtime` partial each partition, combine in an arbitrary order, finalize, and compare against the single-node result.
1. **`seq == par`**: the parallel executor must equal the sequential oracle, as a multiset for unordered relations and exactly for ordered ones.
1. **Differential vs DuckDB** in [`tests/differential/`](https://github.com/stephenoffer/batcher/tree/main/tests/differential), including [`test_diff_operator_matrix.py`](https://github.com/stephenoffer/batcher/blob/main/tests/differential/test_diff_operator_matrix.py), which runs `{collect, spill, iter_batches, distributed}` x `{nulls, empty, one row, duplicates, -0.0/NaN, descending}`.

## Where the code lives

- [`crates/bc-runtime/src/agg/mod.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-runtime/src/agg/mod.rs): `partial`, `combine`, `finalize`, `AggFunc`
- [`crates/bc-runtime/src/keys.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-runtime/src/keys.rs): the one canonical key policy, re-exporting the float rule
- [`crates/bc-arrow/src/float_ident.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-arrow/src/float_ident.rs): `canon_f64_bits` and `float_total_cmp`
- [`crates/bc-runtime/src/agg/spill/mod.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-runtime/src/agg/spill/mod.rs): grace aggregation (the same algebra, bounded)
- [`crates/bc-interp/src/dist.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-interp/src/dist.rs): the distributed primitives
- [`crates/bc-runtime/src/agg/sketch.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-runtime/src/agg/sketch.rs): the HLL and DDSketch aggregate states
- [`crates/bc-sketches/`](https://github.com/stephenoffer/batcher/tree/main/crates/bc-sketches): the mergeable sketches (HLL, DDSketch, KLL, TDigest, Misra-Gries, Bloom), fixed seed

## See also

- {doc}`Architecture </architecture/index>`: the invariant this page is the implementation of.
- `docs/architecture/internals/mathematical_foundations.md` (in the repo, not a site page). It is the v1-era design paper with an errata list at its top, and where it and the code differ the code decides. It covers the algebraic statement and its proofs.
- {doc}`Execution engine </architecture/internals/execution>`: where `partial`/`combine`/`finalize` are called from.
- {doc}`Aggregations </user-guide/analyze/aggregations>`: the surface this algebra is hiding behind.
- {doc}`Scaling benchmarks </benchmarks/results/scaling>`: what bounded per-node memory buys as the cluster grows.
- {doc}`Aggregation internals </architecture/deep-dives/operators/aggregation-internals>`: how `partial` and `combine` actually run.
- {doc}`Morsel parallelism </architecture/deep-dives/operators/morsel-parallelism>`: the scheduling this algebra makes safe.
- {doc}`Distributed scheduling </architecture/deep-dives/distribution/distributed-scheduling>`: the same three functions, mapped over Ray.
