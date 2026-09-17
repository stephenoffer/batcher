# Skewed keys and hostile data shapes

This page describes what happens when the *shape* of the data, rather than its size, decides a query's memory use, and how Batcher keeps those queries inside their budget. Read it when a job that fits your memory budget on paper dies anyway, or when you want to know which shapes still have a hard ceiling.

## What key skew is

A key is *skewed* when a small number of its values cover a large share of the rows. The
canonical source is a sentinel: an ETL job that cannot resolve a dimension lookup writes
`-1`, `0`, or an empty string, and every unresolved row from every day of history shares that
one value.

Skew matters because the standard way to bound a join's memory is to partition both sides by
a hash of the key. Equal keys land in the same partition on both sides, so each partition
pair is an independent join and their union is the whole join. That reasoning is exactly what
makes skew unfixable by partitioning: rows that share a key hash to the same place *by
construction*, so no partition count and no re-hash separates them.

```python
import batcher as bt

# 20,000 unresolved rows share the `-1` sentinel; three rows carry real keys.
facts = bt.from_pydict(
    {
        "customer_id": [-1] * 20_000 + [1, 2, 3],
        "amount": list(range(20_003)),
    }
)
customers = bt.from_pydict({"customer_id": [1, 2, 3], "name": ["ana", "bo", "cy"]})

matched = facts.join(customers, on="customer_id").collect()
print(matched.num_rows)
# 3
```

## How Batcher bounds a skewed join

Batcher holds only the *build* side of a partition in memory. The build side is the smaller
relation, the one a hash table is built over. The probe side is consumed a morsel at a time
and its results are emitted as they are produced, so the probe side's size never enters the
memory bound at all.

That is what makes the join above safe. The `customer_id = -1` rows all land in one
partition, that partition is far larger than any envelope you would set, and it does not
matter: the build side for it is the three-row dimension, and the 20,000 fact rows stream
past it.

The bound holds for every join flavor, including the outer ones. A `RIGHT` or `FULL` join has
to emit build rows that matched nothing, which is a property of the whole probe side rather
than of any one morsel of it. Batcher carries one match mark per build row across the morsels
and emits the remainder once at the end. The marks are one byte per row of the side that is
resident anyway, so they do not change the bound.

Set an envelope and the out-of-core path runs on its own:

```python
from batcher.config import Config, MemoryConfig, config_context

plan = facts.join(customers, on="customer_id")

with config_context(Config().replace(memory=MemoryConfig(max_memory_bytes=1))):
    spilled = plan.collect()

print(spilled.num_rows == matched.num_rows)
# True
```

## Which side Batcher builds on

The planner nominates a build side from estimated cardinalities. Both relations are
materialized before the join runs, so their sizes are facts by then, and Batcher re-checks
the choice against the measured sizes. An inner join whose nominated build side turns out to
be several times larger than its probe side is re-oriented, which is a re-labeling of the
output columns rather than a rewrite.

The re-orientation is restricted to inner joins. The other flavors are not symmetric: their
output depends on which side is allowed to contribute unmatched rows, so exchanging the sides
would change the relation rather than the schedule.

You do not control this directly, and you should not need to. What you can control is the
estimate it corrects: a source with current statistics gives the planner the right answer
the first time, which saves the correction rather than the join.

## Hot keys on a cluster

A distributed join adds a second cost to skew: every row of a hot key crosses the shuffle to one reducer. Batcher finds those keys and spreads them. A heavy-hitters sketch marks a value as hot when it covers `distributed.skew_join_fraction` of the rows, 10% by default. The probe side's hot rows then fan out across several reducers, and the build side's matching rows are copied to each of them. Cold keys hash exactly as before, so the joined relation is unchanged.

The figure shows one hot key on a four-reducer shuffle, before and after salting:

![Two panels show a probe side whose customer_id of -1 covers most rows, shuffled to four reducers. Before salting, the shuffle hashes on the key alone, so every -1 row lands on reducer 0, which carries almost all the load while reducers 1 to 3 hold only cold keys; no bucket count or re-hash separates rows that share a key. After salting, the sketch has marked -1 as hot, and its rows are split across reducers 0, 1 and 2, each also receiving a copy of the matching build-side rows, while reducer 3 still holds only cold keys hashed as before, so the joined relation is unchanged. A key counts as hot when it covers distributed.skew_join_fraction of the rows, 10% by default.](/_static/diagrams/skew_hot_key_salting.svg)

Detection is learned, not repeated. The hot-key list for a join shape is stored in the metadata hub, so the next run salts with no detection pass, and a shape measured as uniform never runs the pass again. `distributed.skew_join_salt` controls the fan-out: `0`, the default, leaves both the decision and the fan-out to the measurement, a positive value forces detection and pins the fan-out, and a negative value turns salting off. {doc}`/architecture/deep-dives/distribution/distributed-scheduling` covers the mechanism.

## Skew in an aggregation

An aggregation is not vulnerable to skew the way a join is. Every row of a group folds into
one accumulator, so a group with a billion rows costs the same state as a group with one. The
memory an aggregation needs scales with the number of *distinct* keys, which is what hash
partitioning does split.

The exception is an aggregate whose state is proportional to its input rather than constant:
`array_agg` collects every value, and an exact `median` or `quantile` has to see the whole
group. Those spill through the same grace machinery, and a single group larger than the
envelope is the case that machinery cannot subdivide.

```python
counts = facts.group_by("customer_id").agg(n=bt.col("amount").count()).collect()
print(sorted(counts.column("n").to_pylist()))
# [1, 1, 1, 20000]
```

## Range joins and the memory envelope

A range join matches on an inequality rather than an equality, so there is no key to
partition on. Batcher decomposes it the other way: a left row's matches depend on the whole
right side and on nothing else about the left, so the left side streams past a resident right
side in chunks sized by the memory envelope.

When the left side fits the envelope, that is a single chunk and the join is the single sweep
it has always been. Only a left side that genuinely exceeds the envelope pays for extra
passes over the right side.

The right side carries a global sort order and cannot be decomposed this way. A right side
larger than the envelope raises `MemoryBudgetExceededError` naming the budget it needed,
rather than letting the process be killed. Put the smaller relation on the right of a range
join.

A right side of at most 32 rows over a large left one is cheaper still, because it needs no
sort at all: the join compares each right row against the left key column directly. A bucket
table, a set of price bands or a handful of date ranges is therefore the least expensive shape
to join on, whatever the left side's size.

Every operator that can't spill refuses the same way. `MemoryBudgetExceededError` is a {py:exc}`bt.ResourceError <batcher.ResourceError>`, so one `except bt.ResourceError` covers all of them:

```python
from batcher._internal.errors import MemoryBudgetExceededError

print(issubclass(MemoryBudgetExceededError, bt.ResourceError))
# True
```

## Diagnosing a skewed key

Profile the key before you tune anything. A key whose top value covers a large share of the
rows is the one to act on:

```python
top = (
    facts.group_by("customer_id")
    .agg(rows=bt.col("amount").count())
    .sort("rows", descending=True)
    .limit(1)
    .to_pydict()
)
print(top["customer_id"][0], top["rows"][0])
# -1 20000
```

If the hot value is a sentinel that matches nothing, filtering it out before the join is
strictly better than any amount of tuning. It removes the rows from the shuffle, the join,
and everything downstream:

```python
resolved = facts.filter(bt.col("customer_id") >= 0).join(customers, on="customer_id")
print(resolved.collect().num_rows)
# 3
```

## Requirements and limitations

- A window function without `PARTITION BY` needs the whole relation ordered at once and
  cannot spill. Over a configured envelope it raises rather than risking the process.
- A single window partition larger than the envelope has the same ceiling. Partitioning by a
  skewed key is the shape to avoid here, because re-partitioning cannot split one partition
  and a ranking needs its partition whole.
- An ASOF join without `by` keys needs one global order over both sides and cannot spill. With
  `by` keys it partitions and spills like an equi join.
- A range join holds its right side whole. Only its left side is decomposed.
- These bounds describe the single-node engine. The distributed executor composes the same
  mergeable primitives, so the same shapes are the hard ones there.

## See also

- {doc}`Performance and memory <performance>`: the memory envelope and the rest of the levers.
- {doc}`Reading query plans <explain-plans>`: finding which operator is actually costing you.
- {doc}`large-tables`: bucket counts, and why more buckets don't fix a hot key.
- {doc}`/user-guide/analyze/joins`: the join flavors and how to choose one.
- {doc}`/user-guide/operate/running/troubleshooting`: what a query that dies looks like from
  the outside.
