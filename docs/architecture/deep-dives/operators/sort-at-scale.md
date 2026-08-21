# Sorting at scale

This page covers how a sort behaves as the cluster and the data grow: which phases scale with
the workers, and how the engine adapts when the data is already ordered, holds few distinct
values, or is dominated by one key. {doc}`Sort internals </architecture/deep-dives/operators/sort-internals>` covers the algorithms
themselves.

Both questions have the same answer underneath. A distributed sort is a range partition
followed by independent per-range sorts, so what decides whether it scales is whether the
ranges stay even. What decides that is the shape of the data, not the size of the cluster.

## How the distributed sort scales

A distributed sort is four phases, and the cost of each is worth stating separately, because
"scales with nodes" is a claim about the phases that are *not* per-worker constant.

| Phase | Per worker | Notes |
|---|---|---|
| Sample | `rows / W` | Each worker samples its own split; nothing is read on the driver. Skipped entirely when the shape has been sampled before (`dist/sort_boundaries.py`). |
| Merge boundaries | n/a | On the driver. |
| Range-partition and publish | `rows / W` | The map side, over the credit-bounded Flight shuffle. |
| Reduce (sort a bucket) | `rows / P` | `P` reducers, each sorting its own range. |

Two terms are not per-worker constant, and both are bounded rather than absent:

The **boundary merge** is serial on the driver. What saves it is that the pooled sample size
is `samples_per_bucket · P`, a function of the bucket count rather than of the worker count,
because `sample_probs` scales each worker's grid *down* as the fleet grows. So the driver sorts
a few thousand values however wide the cluster is.

The **shuffle** opens `W · P` streams. The bytes are `rows` in total however they are divided,
so this is a connection count rather than a data volume, and it is what every one-round shuffle
costs.

What is emphatically *not* a term is the result. A sort is row-preserving, so `collect()` pulls
the whole relation through the driver. `write` does not: the shard streams on the worker that
produced it, a chunk at a time. A sort feeding a write therefore has no `O(rows)` driver
term at all, which is the shape a large sort actually has.

:::{warning}
The one thing that genuinely broke scaling was **skew**, and it did so silently. See the next
section. A value holding share `f` of the rows kept `f·N` of them on one reducer no matter how
many workers were added, so the busiest bucket did not move while every other one shrank.
:::

## Adapting to the data

The sort has no single algorithm, and which one runs is decided from the *data* rather than
from the query. Four shapes get their own treatment, and all four now apply to every key
family rather than only to numbers or only to text.

| The data is | What happens | Where |
|---|---|---|
| Already in key order | The permutation is the identity, found in one pass | `already_ordered` |
| A handful of distinct values | Ranked and counted, no comparisons at all | `lowcard::rank_part_of`, `rank_sort_live` |
| Narrow fixed-width keys | Packed into one `u64` and radix-sorted | `radix_sort_live` |
| Dominated by one value | That value gets a bucket of its own, spread across several reducers | `plan_hot_split` |

The last row is the one that decides whether a distributed sort scales. A range partition must
keep equal keys together, because the result is the ordered concatenation of the buckets, so a
value holding share `f` of the rows pins `f·N` of them on a single reducer *however wide the
shuffle is*. Adding workers shrinks every other reducer's share and leaves that one alone, so
the overload grows with the cluster and the speedup is capped at `1/f`.

Isolating the hot value needs its immediate successor as a boundary, and for a long time that
was read as "numeric keys only, because a string has no cheap successor". That was wrong. A
byte key's successor is the value with a `\x00` appended, and nothing sorts between them. A
value above it either has it as a proper prefix, so its next byte is at least `\x00`, or it
differs inside its bytes and is above both. The float path's `nextafter` is the *nearest
representable* value. This one is the successor exactly.

Measured over 600,000 rows with 40% on one value, the busiest bucket used to sit at 240,000
rows at 8, 16 and 32 buckets alike, an overload of 3.2x, 6.4x and 12.8x. It now tracks the
even share at 1.00x throughout.

:::{note}
The rearrangement is sound only because every row it moves *ties* on the key, so their relative
order is free, subject to the one constraint that makes a limited sort match single-node:
concatenating the sub-buckets in order must reproduce mapper order. A skew bug here
does not lose keys. It returns the right keys carrying the wrong rows, which an
order-independent assertion reads as a pass, so the tests compare the full row multiset.
:::

## See also

- {doc}`Sort internals </architecture/deep-dives/operators/sort-internals>`: the five sort paths and the permutation they must agree on.
- {doc}`Mergeable algebra </architecture/deep-dives/operators/mergeable-algebra>`: why one core and one cluster run the same operator.
- {doc}`Morsel parallelism </architecture/deep-dives/operators/morsel-parallelism>`: where the per-range sorts get their cores.
- {doc}`Distributed scheduling </architecture/deep-dives/distribution/distributed-scheduling>`: the Ray scheduling these phases run on.
- {doc}`Flight shuffle </architecture/deep-dives/distribution/shuffle-flight>`: the transport the map side publishes into, and its credit-based backpressure.
- {doc}`Sorting </user-guide/transform/rows/sorting>`: the API, and what to reach for when a sort is slow.

