# Physical properties: ordering and partitioning

This page describes the two physical properties Batcher tracks through a plan, what each one lets the optimizer remove, and the rules that decide when a property is safe to claim.

Cardinality estimation answers "how many rows". Physical properties answer "in what shape". A relation that already arrives in the order you asked for needs no sort, and a relation whose rows are already grouped on the right key needs no shuffle.

The property algebra is in `batcher/kyber/properties.py`, the vocabulary in `plan/stats.py`, and the propagation in `batcher/kyber/stats/estimator.py`.

## What an ordering is

An ordering is a sequence of `SortOrder` keys, each naming a column, its direction, and where its nulls sit. You can ask any dataset what ordering it is known to be in:

```python
import batcher as bt

ds = bt.from_pydict({"ts": [3, 1, 2], "v": [30, 10, 20]})
print(ds.sort("ts", descending=True).meta.sorted_by())
# (SortOrder(column='ts', descending=True, nulls_first=False),)
```

All three parts matter. `ts ASC` and `ts DESC` don't satisfy each other, and two orderings that differ only in null placement interleave their rows differently. Recording the direction is what lets the most common analytic shape, `ORDER BY ts DESC`, carry an ordering at all. An empty result means "no recorded ordering", not "unordered".

## What the ordering removes

A sort is redundant when its input already delivers an ordering that satisfies it, meaning the delivered keys are a prefix extension of the required ones: rows sorted by `(a, b)` are also sorted by `(a,)`.

```python
import batcher as bt

ds = bt.from_pydict({"ts": [3, 1, 2], "v": [30, 10, 20]})
once = ds.sort("ts", descending=True)
twice = once.sort("ts", descending=True)
print(once.collect().to_pydict() == twice.collect().to_pydict())  # True
print(twice.explain().count("sort  ["))                           # 1
```

The plan the engine runs holds one sort, not two. The rule declines whenever the claim isn't exact, so an ascending sort over a descending input keeps both sorts:

```python
import batcher as bt

ds = bt.from_pydict({"ts": [3, 1, 2], "v": [30, 10, 20]})
print(ds.sort("ts", descending=True).sort("ts").collect().to_pydict()["ts"])  # [1, 2, 3]
```

### A top-N becomes a limit

When the input already delivers the ordering a top-N asks for, its first `n` rows *are* the top `n`, so the heap collapses to a limit:

```text
ORDER BY ts DESC LIMIT 10   over a table stored newest-first

  before:  sort(limit=10) <- scan          reads every row, keeps a heap of 10
  after:   limit(10)      <- scan          reads 10 rows
```

That is the standard recent-events query against a lakehouse table with a descending sort key. The rewrite falls out of sort elimination: the query reaches the rewrite phase as a `Limit` above a plain `Sort`, so removing the sort leaves the limit on the scan. `ORDER BY ts ASC LIMIT 10` over the same table keeps its sort.

When the input order isn't known, the limit still fuses into the sort as a top-N and is pushed to the scan as a hint:

```python
import batcher as bt

ds = bt.from_pydict({"ts": [3, 1, 2], "v": [30, 10, 20]})
print(ds.sort("ts", descending=True).limit(2).explain())
```

```text
query plan (planned)                                 2 operators
────────────────────────────────────────────────────────────────
OPERATOR             ESTIMATE  NOTES
sort  [top 2 by ts]     est≈2  (exact)
└─ scan  [source 0]     est≈3  (exact)  pushed[top 2 by ts desc]
```

Null placement is compared exactly, with one relaxation: when a column is *proven* to hold no nulls, `NULLS FIRST` and `NULLS LAST` describe the same row order and either satisfies the other. An estimated null count doesn't qualify.

## Which operators carry an ordering

An operator carries its input's ordering when it can't move a row relative to another row:

| Operator | Carries the ordering | Why |
|---|---|---|
| `Filter` | Yes | Dropping rows from a sorted relation leaves it sorted. |
| `Project` | Yes, renamed | The prefix ends at the first order key the projection does not carry forward as a bare column. |
| `Limit` | Yes | A prefix of a sorted relation is sorted. |
| `Sample` | Yes | Rows are only ever dropped, and the sampler preserves relative order in both modes. |
| `Unnest` | Yes, truncated | Each input row becomes several output rows in place. The exploded column itself ends the prefix. |
| `Window` | Yes | It appends columns and moves no row. |
| `Aggregate` | No | A hash aggregate emits groups in no defined order. |
| `Join` | No | A hash join emits rows in build and probe order, not input order. |
| `Union` | No | The branches concatenate, so branch order dominates. |

A computed sort key, such as `lower(name)`, is never carried, because no column holds that value.

## What a partitioning is

A partitioning names the key set whose equal values are guaranteed to share a worker, so a distributed plan can skip a shuffle it doesn't need. Partitioning and ordering contain in *opposite* directions:

- an **ordering** satisfies a requirement when the delivered keys are a prefix *extension* of the required ones;
- a **partitioning** satisfies a grouping requirement when the delivered keys are a *subset* of the required ones.

Rows partitioned by `hash(a)` keep every `(a, b)` group whole, so partitioning on `(a)` satisfies grouping by `(a, b)`. Partitioning on `hash(a, b)` does **not** satisfy grouping by `(a)`, because one `a` group straddles several buckets.

A partitioning can also come from storage: a table partitioned on disk hands each partition's rows to one worker, so the distributed scheduler supplies it as `clustered_on` and `satisfies` treats it the same way. An empty partitioning guarantees nothing, and leaving one unclaimed costs at most an extra shuffle.

![Ordering and partitioning, which do not work alike. An ordering travels with the plan: Scan orders establishes the ordering (o_date) from a proved footer order, Filter preserves it, Project preserves it under the new column name as (day), and Aggregate destroys it, so an empty ordering leaves the far side and a later sort has to run. A partitioning is never carried, only recomputed: rows pass from a Join on k through a Filter to an Aggregate on (k, x) with no property riding along, and the point of decision, dist scheduling or the cost model, walks the plan again there and then. Nothing stores a partitioning and there is no Exchange node to enforce one. In that plan the delivered (k) sits inside the required (k, x), so the shuffle is skipped. The two contain in opposite directions: an ordering satisfies a requirement when it is longer, because rows sorted by (a, b) are also sorted by (a) while (a) alone does not satisfy (a, b); a partitioning satisfies one when it is a subset, because partitioning on (a) keeps every (a, b) group whole while partitioning on (a, b) does not keep an (a) group whole. Getting it backwards drops a sort that was needed or skips a shuffle that was not optional, and a wrong claim about either is a wrong answer, not a slow one.](/_static/diagrams/physical_properties.svg)

## Claims are proved, never guessed

Every other statistic in the planner is a bound, so being wrong about it makes a plan slower. An ordering claim lets the optimizer *delete* a sort, so a wrong claim would return rows in the wrong order. Two habits follow, and the test suite enforces both:

- claim a property only when it is *proved*, never when it is merely likely;
- test an ordering with an order-*sensitive* assertion, because the default comparison in [`tests/differential/`](https://github.com/stephenoffer/batcher/tree/main/tests/differential) is order-independent by design.

`eliminate_sort_before_aggregate` is the sound form of sort removal for consumers whose own output order is unspecified: it drops a sort beneath a group-by, looking through an intervening sample, because the aggregate's output order is undefined either way.

## Where a source ordering comes from

A connector can declare the ordering its data is stored in, and Batcher then treats a sort on that prefix as free. For Parquet the declaration is proved in `batcher/io/stats/sortedness.py`, and all three conditions must hold:

1. Every row group of every file declares the same leading sorting column running the same direction.
1. Row groups are ordered within each file, checked against their own min and max bounds.
1. Files are ordered across the dataset, in the order the scan reads them.

Both directions are provable and both are claimed. A missing statistic, a null in the key, or an unordered pair drops the claim, and declining costs only the sort that was going to run anyway.

## See also

- {doc}`Sort internals </architecture/deep-dives/operators/sort-internals>`: how the sort that does survive is executed.
- {doc}`The plan IR </architecture/deep-dives/query/plan-ir>`: the contract the optimized plan is lowered to.
- {doc}`Query lifecycle </architecture/deep-dives/query/query-lifecycle>`: where in a query the optimizer runs.
- {doc}`Cardinality estimation </architecture/deep-dives/adaptive/cardinality-estimation>`: the other half of what the optimizer knows about a relation.
- {doc}`Partition-aware planning </architecture/deep-dives/distribution/partition-aware-planning>`: how the distributed path uses a partitioning to skip a shuffle.
