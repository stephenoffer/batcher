# Sorting at scale

This page covers how a sort behaves as the cluster and the data grow: which phases scale with the workers, and how the engine adapts when the data is already ordered, holds few distinct values, or is dominated by one key. {doc}`Sort internals </architecture/deep-dives/operators/sort-internals>` covers the algorithms themselves.

A distributed sort is a range partition followed by independent per-range sorts, so whether it scales depends on whether the ranges stay even, which is a property of the data rather than the cluster size. From the API it's one call, and a sort's order is a guarantee on one node or several:

```python
# docs: skip
import batcher as bt

n = 200_000
ds = bt.from_pydict({"k": [7 if i % 5 < 2 else i for i in range(n)], "i": list(range(n))})
q = ds.sort("k", "i")  # 40% of the rows share k = 7
one_node = q.collect(distributed=False)
four_workers = q.collect(distributed=True, num_workers=4)  # needs a Ray cluster
```

## How the distributed sort scales

A distributed sort runs in four phases:

| Phase | Per worker | Notes |
|---|---|---|
| Sample | `rows / W` | Each worker samples its own split; nothing is read on the driver. Skipped entirely when the shape has been sampled before (`dist/sort_boundaries.py`). |
| Merge boundaries | n/a | On the driver. |
| Range-partition and publish | `rows / W` | The map side, over the credit-bounded Flight shuffle. |
| Reduce (sort a bucket) | `rows / P` | `P` reducers, each sorting its own range. |

Two terms aren't per-worker constant, and both are bounded:

- The **boundary merge** is serial on the driver, but the pooled sample is `samples_per_bucket · P`, a function of the bucket count, because `sample_probs` scales each worker's grid down as the fleet grows. The driver sorts a few thousand values however wide the cluster is.
- The **shuffle** opens `W · P` streams. The bytes total `rows` however they're divided, so this is a connection count, not a data volume.

A sort feeding a `write` has no `O(rows)` driver term at all: each shard streams out on the worker that produced it, a chunk at a time. Only `collect()` pulls the relation through the driver.

## Adapting to the data

Which algorithm runs is decided from the data, not the query. Four shapes get their own treatment, for every key family:

| The data is | What happens | Where |
|---|---|---|
| Already in key order | The permutation is the identity, found in one pass | `already_ordered` |
| A handful of distinct values | Ranked and counted, no comparisons at all | `lowcard::rank_part_of`, `rank_sort_live` |
| Narrow fixed-width keys | Packed into one `u64` and radix-sorted | `radix_sort_live` |
| Dominated by one value | That value gets a bucket of its own, spread across several reducers | `plan_hot_split` |

The last row decides whether a distributed sort scales. A range partition must keep equal keys together, so a value holding share `f` of the rows would pin `f·N` of them on one reducer however wide the shuffle is, capping the speedup at `1/f`. `plan_hot_split` gives the hot value its own bucket, bounded by its immediate successor, and spreads that bucket across several reducers. Measured over 600,000 rows with 40% on one value, the busiest bucket tracks the even share at 1.00x at 8, 16 and 32 buckets.

:::{dropdown} Why the split is exact for every key type
Isolating a hot value needs its immediate successor as a boundary. For a float that's `nextafter`. For a byte key it's the value with a `\x00` appended, and nothing sorts between them: a value above it either has it as a proper prefix, so its next byte is at least `\x00`, or differs inside its bytes and is above both. So the hot-value split works for text and binary keys as well as numbers.

The rearrangement is sound because every row it moves ties on the key, so their relative order is free, subject to one constraint: concatenating the sub-buckets in order must reproduce mapper order, which is what makes a limited sort match single-node. The tests compare the full row multiset, not just the keys, because a skew bug would return the right keys carrying the wrong rows.
:::

## See also

- {doc}`Sort internals </architecture/deep-dives/operators/sort-internals>`: the five sort paths and the permutation they must agree on.
- {doc}`Mergeable algebra </architecture/deep-dives/operators/mergeable-algebra>`: why one core and one cluster run the same operator.
- {doc}`Morsel parallelism </architecture/deep-dives/operators/morsel-parallelism>`: where the per-range sorts get their cores.
- {doc}`Distributed scheduling </architecture/deep-dives/distribution/distributed-scheduling>`: the Ray scheduling these phases run on.
- {doc}`Flight shuffle </architecture/deep-dives/distribution/shuffle-flight>`: the transport the map side publishes into, and its credit-based backpressure.
- {doc}`Sorting </user-guide/transform/rows/sorting>`: the API, and what to reach for when a sort is slow.
