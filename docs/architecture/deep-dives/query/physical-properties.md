# Physical properties: ordering and partitioning

This page describes the two physical properties Batcher tracks through a plan, what each one
lets the optimizer remove, and the rules that decide when a property is safe to claim.

Cardinality estimation answers "how many rows". Physical properties answer the other
question a mature optimizer needs: "in what shape". A relation that already arrives in the
order you asked for needs no sort, and a relation whose rows are already grouped on the
right key needs no shuffle. Both are pure savings, and both are invisible unless the
property is propagated.

The code is `batcher/kyber/properties.py` for the property algebra, `plan/stats.py` for the
vocabulary, and `batcher/kyber/stats/estimator.py` for the propagation.

## What an ordering is

An ordering is a sequence of `SortOrder` keys. Each key names a column, whether it descends,
and where its nulls sit:

```python
from batcher.plan.stats import SortOrder

SortOrder("ts", descending=True, nulls_first=False)
```

All three parts matter. `ts ASC` and `ts DESC` are different orderings and neither satisfies
the other, so the direction cannot be dropped. Null placement decides where the null rows
land, so two orderings differing only there interleave their rows differently.

Recording the direction is what makes the property pay for itself. An ordering restricted to
ascending keys cannot describe `ORDER BY ts DESC`, which is how nearly every recent-first
query is written, so the most common ordered shape in analytics used to deliver no ordering
at all and every consumer of the property was blind to it.

You can ask any dataset what ordering it is known to be in:

```python
import batcher as bt

ds = bt.from_pydict({"ts": [3, 1, 2], "v": [30, 10, 20]})
print(ds.sort("ts", descending=True).meta.sorted_by())
```

An empty result means "no recorded ordering", which is not the same as "unordered". Only a
declared or derived ordering is tracked.

## What the ordering removes

A sort is redundant when its input already delivers an ordering that satisfies it. An
ordering satisfies a requirement when it is a prefix extension of it: rows sorted by
`(a, b)` are also sorted by `(a,)`, never the reverse.

```python
import batcher as bt

ds = bt.from_pydict({"ts": [3, 1, 2], "v": [30, 10, 20]})
once = ds.sort("ts", descending=True)
twice = once.sort("ts", descending=True)
print(once.collect().to_pydict() == twice.collect().to_pydict())
```

The second sort does no work. The plan the engine runs holds one sort, not two.

The rule declines whenever the claim is not exact. An ascending sort over a descending input
keeps both sorts, because neither ordering satisfies the other:

```python
import batcher as bt

ds = bt.from_pydict({"ts": [3, 1, 2], "v": [30, 10, 20]})
print(ds.sort("ts", descending=True).sort("ts").collect().to_pydict()["ts"])
```

### A top-N becomes a limit

The same reasoning removes more than a sort. When the input already delivers the ordering a
top-N asks for, its first `n` rows *are* the top `n`, already in the right order, so the
whole heap collapses to a limit:

```text
ORDER BY ts DESC LIMIT 10   over a table stored newest-first

  before:  sort(limit=10) <- scan          reads every row, keeps a heap of 10
  after:   limit(10)      <- scan          reads 10 rows
```

That is the standard recent-events query against a lakehouse table with a descending sort
key, and it is the case that motivated recording the direction in the first place. An
ordering that could only describe ascending keys never matched it.

No separate top-N rule does this, and one would not fire if it existed. The query reaches the
rewrite phase as a `Limit` sitting *above* a plain `Sort` -- a sort only acquires its own
`limit` in a later phase -- so eliminating the sort is what leaves the limit on the scan.

The rewrite declines whenever the ordering is not an exact prefix match, so
`ORDER BY ts ASC LIMIT 10` over the same newest-first table keeps its sort.

### Null placement and columns that hold no nulls

Null placement is compared exactly, with one relaxation. When a column is *proven* to hold no
nulls there is no null row whose position could distinguish the two spellings, so
`NULLS FIRST` and `NULLS LAST` describe the same row order and either satisfies the other.
The proof has to be exact. An estimated null count does not qualify, and an unknown one
certainly does not.

## Which operators carry an ordering

An operator carries its input's ordering when it cannot move a row relative to another row.

| Operator | Carries the ordering | Why |
|---|---|---|
| `Filter` | Yes | Dropping rows from a sorted relation leaves it sorted. |
| `Project` | Yes, renamed | A projection reorders nothing. The prefix ends at the first order key the projection does not carry forward as a bare column. |
| `Limit` | Yes | A prefix of a sorted relation is sorted. |
| `Sample` | Yes | Rows are only ever dropped, and the sampler preserves relative order in both modes. |
| `Unnest` | Yes, truncated | Each input row becomes several output rows in place, which introduces ties rather than breaking the order. The exploded column itself ends the prefix. |
| `Window` | Yes | It appends columns and moves no row. |
| `Aggregate` | No | A hash aggregate emits groups in no defined order. |
| `Join` | No | A hash join emits rows in build and probe order, not input order. |
| `Union` | No | The branches concatenate, so branch order dominates. |

A computed sort key is never carried, whatever the operator. Sorting by `lower(name)` orders
the relation by a value no column holds, so no consumer could name it.

## What a partitioning is

A partitioning names the key set whose equal values are guaranteed to share a worker. It
exists so a distributed plan can skip a shuffle it does not need.

Partitioning and ordering contain in *opposite* directions, which is the most error-prone
thing on this page:

- an **ordering** satisfies a requirement when the delivered keys are a prefix *extension* of
  the required ones;
- a **partitioning** satisfies a grouping requirement when the delivered keys are a *subset*
  of the required ones.

Rows partitioned by `hash(a)` keep every `(a, b)` group whole, because equal `(a, b)` implies
equal `a` and therefore one bucket. So partitioning on `(a)` satisfies grouping by `(a, b)`.
Partitioning on the superset `hash(a, b)` does **not** satisfy grouping by `(a)`: two rows
sharing `a` but differing in `b` hash to different buckets, the `a` group straddles them, and
a reducer that skipped the shuffle emits a partial group. That is a wrong answer, not a slow
one.

An empty partitioning guarantees nothing and satisfies only an empty requirement. Leaving a
partitioning unclaimed costs at most an unnecessary shuffle, so the safe answer is always to
claim nothing.

## Why a wrong claim is worse than no claim

Every other statistic in the planner is a bound, so being wrong about it makes a plan slower.
An ordering claim is different: the optimizer *deletes* a sort on the strength of it, and the
query then returns rows in the wrong order. Nothing raises, because a wrong order is not an
error, and an order-independent comparison cannot see it.

That is not hypothetical. A rule once rewrote `Sample(Sort(x))` to `Sample(x)` on the grounds
that the sampled multiset does not depend on input order. The multiset argument is correct
and it is not sufficient, because the multiset is not the only observable:
`ds.sort("a").sample(fraction=0.3)` returned its rows in scan order. Every check on that rule
used the order-independent comparison, so the one property that broke was the one nothing
compared.

Two habits follow, and both are enforced in the test suite:

- claim a property only when it is *proved*, never when it is merely likely;
- test an ordering with an order-*sensitive* assertion, because the default comparison in
  `tests/differential/` is order-independent by design.

The sound form of that rewrite matches the consumer whose own output order is unspecified.
`eliminate_sort_before_aggregate` removes a sort beneath a group-by, looking through an
intervening sample, because an aggregate's output order is undefined in the original plan and
in the rewritten one alike. Nothing observable changes.

## Where a source ordering comes from

A connector can declare the ordering its data is stored in, and Batcher then treats a sort on
that prefix as free. For Parquet the declaration is proved rather than trusted, in
`batcher/io/stats/sortedness.py`. All three conditions must hold:

1. every row group of every file declares the same leading sorting column running the same
   direction;
2. row groups are ordered within each file, checked against their own min and max bounds;
3. files are ordered across the dataset, in the order the scan reads them.

Both directions are provable and both are claimed. Any missing statistic, any null in the
key, or any unordered pair drops the claim, and the cost of declining is a sort that was
going to happen anyway.

## See also

- {doc}`Sort internals </architecture/deep-dives/operators/sort-internals>`: how the sort that
  does survive is executed.
- {doc}`The plan IR </architecture/deep-dives/query/plan-ir>`: the contract the optimized plan
  is lowered to.
- {doc}`Query lifecycle </architecture/deep-dives/query/query-lifecycle>`: where in a query the
  optimizer runs.
