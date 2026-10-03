# Sort internals

Sort is the one operator whose *order* is the answer. Every hash-based operator produces an unordered relation, so a bug there shows up as a wrong row. A sort bug shows up as rows in the wrong order, which an order-independent assertion can't see. That's why this operator carries more determinism machinery than any other. This page covers the five sort paths, the tie-break that makes them agree, top-N fusion, and the gather that dominates most sorts.

Every path breaks ties on the original row index, so tied rows keep their input order in either direction, and the out-of-core path returns the identical sequence:

```python
import batcher as bt
from batcher.config import Config, MemoryConfig, config_context

t = bt.from_pydict({"k": [2, 1, 2, 1, None], "row": [0, 1, 2, 3, 4]})
print(t.sort("k").to_pydict())
# {'k': [1, 1, 2, 2, None], 'row': [1, 3, 0, 2, 4]}
print(t.sort("k", descending=True).to_pydict())
# {'k': [2, 2, 1, 1, None], 'row': [0, 2, 1, 3, 4]}

ds = bt.from_pydict({"x": [i * 7919 % 1000 for i in range(5000)], "i": list(range(5000))})
q = ds.sort("x", "i", descending=True)
with config_context(Config().replace(memory=MemoryConfig(max_memory_bytes=1))):
    print(q.to_pydict() == ds.sort("x", "i", descending=True).to_pydict())  # spilled == in-memory
# True
```

:::{important}
The engine has five sort paths, and all five must produce the **identical permutation**, not merely a correctly sorted one. The parallel and spilling paths each sort a slice of the input and concatenate the results, so two paths that ordered tied rows differently would return different relations from the same query.
:::

| Path | Taken when | Code |
|---|---|---|
| LSD radix | a single integer, temporal or float key, a full sort, no `NaN` in the column | `ops/radix_sort/` |
| Composite packed key | two to eight integer, temporal or float keys whose measured value ranges fit 96 bits between them | `ops/radix_sort/packed.rs` |
| Stable byte-key sort | a `Utf8`, `LargeUtf8`, `Binary`, `LargeBinary` or `FixedSizeBinary` key | `ops/byte_sort.rs` |
| Parallel sample-sort | above 2^17 rows, a full sort, leading key of type float / integer / temporal / text / binary | `ops/sample_sort/` |
| External merge sort | the input exceeds the memory envelope | `ops/external_sort.rs` |

```text
   sort_indices(keys, batch)
        ├─ input exceeds the memory envelope? ──────► external merge sort
        ├─ > 2^17 rows, full sort, leading key
        │  is int / float / temporal / bytes? ──────► parallel sample-sort
        ├─ single text or binary key? ──────────────► stable byte-key sort
        ├─ single fixed-width key, no NaN? ─────────► LSD radix sort
        ├─ 2-8 numeric / temporal keys whose
        │  measured ranges fit 96 bits? ────────────► composite packed-key sort
        └─ otherwise ───────────────────────────────► row-encoded stable comparison sort

   every path breaks ties on the original row index, so the permutation is unique.
```

## The permutation and its tie-break

`sort_indices` ([`crates/bc-interp/src/ops/mod.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-interp/src/ops/mod.rs)) evaluates the key expressions and builds a `UInt32Array` permutation, and `sort_batch` `take`s the whole batch through it. Arrow's comparison sorts aren't stable, so every path breaks ties on the original row index: the radix paths through stability or by carrying it inside the packed word, the row-encoded fallback (`bc_arrow::row_sort`) in its comparator, and the last-resort `lexsort` as a final ascending key. The slice a parallel range sorts is always gathered in ascending original-row order, so a slice-local index preserves the input's tie order.

## Natural runs come first

Real data is often partly in order: files each written sorted, an append-only log, a `UNION ALL` of sorted sources. Before either radix path runs, `ops/run_sort.rs` finds the key's maximal ordered runs, sorts only what lies between them, and merges pairwise in `log2(k)` parallel rounds. The detection strides like DuckDB's vergesort, jumping `n / log2(n)` positions at a time, so failing to find a run costs about `3 * log2(n)` comparisons. That's what lets it run unconditionally. Unlike vergesort, a descending run is reversed only when it's strictly descending, which keeps the sort stable.

:::{dropdown} Measured effect of run detection
Six million rows with a `float64` payload, two builds differing only in whether detection runs:

| key structure | detection off | detection on | |
|---|---|---|---|
| random (the control) | 36.1 ms | 34.2 ms | 1.06x |
| strictly descending | 23.6 ms | 17.8 ms | **1.32x** |
| two sorted halves concatenated | 23.1 ms | 18.4 ms | **1.26x** |

Input with more than sixty-four runs, or whose runs cover less than half the rows, is handed back untouched.
:::

## The fixed-width paths

**LSD radix** (`ops/radix_sort/`) sorts an order-preserving `u64` transform of a single key: sign-flipped for signed integers, bit-inverted for descending, an IEEE-order transform matching Arrow's `total_cmp` for floats. It's O(n·w) against O(n log n) and produces the identical relation. Above 2^18 rows a float column sorts `(key, row)` pairs instead of scattering an index, since the key array has left cache. A `NaN`, a string or boolean key, or a top-N declines to the comparison sort.

**Composite packed key** (`packed_multi_sort_indices`) handles a multi-key `ORDER BY` such as `ORDER BY o_orderdate, o_shippriority`. It measures each key's live value range, gives it only the bits that range needs, and packs the tuple into one word, most-significant key first, with the row position in the low 32 bits. Comparing packed words as integers is lexicographic order, and ties keep input order. Up to 32 key bits fit a `u64`; up to 96 take a `u128`, which is what admits floats. Direction is folded into each field, so mixed directions still sort ascending:

```python
orders = bt.from_pydict(
    {
        "orderdate": [19_950_301, 19_950_302, 19_950_301, 19_950_302],
        "shippriority": [0, 1, 1, 0],
        "revenue": [10.0, 20.0, 30.0, 40.0],
    }
)
print(orders.sort("orderdate", "shippriority", descending=[False, True]).to_pydict())
# {'orderdate': [19950301, 19950301, 19950302, 19950302], 'shippriority': [1, 0, 1, 0],
#  'revenue': [30.0, 10.0, 20.0, 40.0]}
```

:::{dropdown} Packed-key details and measurements
The narrowing is the idea DuckDB calls compressed materialization (`src/optimizer/compressed_materialization/compress_order.cpp`), but Batcher measures the range on the rows in hand, so it needs no statistics and narrows intermediates no catalog describes. Nulls are encoded inside their field (lowest value under `nulls_first`, highest otherwise), and a constant column takes zero bits. It declines on a string or boolean key, under 64 rows, over eight keys, or past 96 bits; a range measured over a 4,096-row prefix is enough to reject a key that won't fit.

Measured on 8M rows, best of three interleaved runs against the comparison sort:

| Shape | Comparison sort | Packed | |
|---|---|---|---|
| `ORDER BY <date>, <priority>` (12,000 distinct pairs) | 1,878 ms | 625 ms | **3.0x** |
| `ORDER BY <int>, <int>` (3 M distinct pairs) | 112 ms | 59 ms | **1.9x** |
| `ORDER BY <int> DESC, <int>` | 90 ms | 60 ms | **1.5x** |
| `ORDER BY <int>, <int>, <int>` (narrow) | 133 ms | 71 ms | **1.9x** |
:::

## Stable byte-key sort

`ops/byte_sort.rs` sorts `Utf8`, `LargeUtf8`, `Binary`, `LargeBinary` and `FixedSizeBinary` as one sort, because Arrow orders all five by `memcmp` on the value bytes. Nulls group by `nulls_first` in input order, and descending inverts only the key comparison, never the tie-break. Four strategies are tried cheapest first, all producing the same permutation:

1. **Already ordered.** One pass, and the permutation is the identity.
1. **Rank.** For a few thousand distinct values, hash each row to a dense id, order the distinct values, and place every row by counting.
1. **Radix.** A key of eight bytes or fewer that a zero-padded pack orders exactly becomes one `u64` for the integer radix.
1. **Packed-prefix comparison.** Otherwise, the first eight bytes ride inline as a `u64`, so most comparisons are a register compare.

```python
import pyarrow as pa

records = pa.table(
    {
        "key": pa.array([b"\x02\x00", b"\x00\xff", b"\x01\x7f"], type=pa.binary(2)),
        "payload": pa.array([b"c" * 8, b"a" * 8, b"b" * 8], type=pa.binary()),
    }
)
print(bt.from_arrow(records).sort("key").to_pydict()["key"])
# [b'\x00\xff', b'\x01\x7f', b'\x02\x00']
```

A short fixed-width key over a wide payload (a hash, a UUID, the 10-byte-key/90-byte-payload record of [CloudSort](https://sortbenchmark.org/)) is the canonical large-sort shape. Against DuckDB on the same Arrow input, `python benchmarks/cloudsort.py` measures it at 7.2x to 21.3x, growing with scale:

| Records | Case | Batcher | DuckDB | Ratio |
|---|---|---|---|---|
| 1M | `binary(10)`, key only | 16.4 ms | 155.0 ms | 9.4x |
| 1M | `binary(10)` + `binary(90)` payload | 35.0 ms | 366.7 ms | 10.5x |
| 16M | `binary(10)`, key only | 171.7 ms | 2220.7 ms | 12.9x |
| 16M | `binary(10)` + `binary(90)` payload | 453.0 ms | 9369.5 ms | 20.7x |

## Parallel sample-sort

`ops/sample_sort/`, above 2^17 rows. Sample ~8,192 rows to estimate quantile boundaries, range-partition the rows by the leading key, and sort each range in parallel. The ranges are globally ordered, so the sorted relation is the ranges in key order, with no final merge and no concatenation. Multi-key sorts bucket by the leading key and sort each range by the full key list. Temporal keys (`Date32`, `Date64`, `Timestamp`, `Time64`, `Duration`) route as `i64`. This is the single-node form of the distributed range sort, built on the same `bc_runtime::shuffle` range partitioners.

```text
   sample ~8,192 rows ──► quantile boundaries   b0 < b1 < b2
   route each row to a range by its leading key   (row INDICES only, no payload copy)
   ┌──────────┬──────────┬──────────┬──────────┐
   │ range 0  │ range 1  │ range 2  │ range 3  │  globally ordered relative to each other
   └────┬─────┴────┬─────┴────┬─────┴────┬─────┘
     sort keys  sort keys  sort keys  sort keys   in parallel, on the key columns alone
        └──────────┴────┬─────┴──────────┘
         gather every column ONCE, per range: the ranges, in order, ARE the result
```

How a distributed sort scales and adapts to skew is on {doc}`Sorting at scale </architecture/deep-dives/operators/sort-at-scale>`.

## External merge sort

`ops/external_sort.rs`, when the input exceeds the memory envelope. Each morsel is sorted into a run and spilled, then the runs merge through a bounded-fan-in k-way merge over a `BinaryHeap` of encoded rows. Peak memory is O(`sort_merge_fanin` morsels), 16 by default, regardless of input size, and the result equals an in-memory `sort_batch` over the whole input. Spill files are Arrow IPC through the same `DiskSpillStore` the aggregate uses.

![External sorting in two phases. If the sort fits its memory envelope, parallel_sort_batch runs with no disk at all. If it does not, pass 0 accumulates morsels into a run of at most a quarter of the operator budget capped at 64 MiB, sorts it, and writes it to one Arrow IPC file that is closed immediately, dropping each input batch as it goes so the relation is never all resident. Passes 1 onward then merge at most 16 runs at a time through a min-heap over each run's head row, with one batch per reader resident, writing one longer run per group and repeating while more than one run remains; the final pass streams the sorted rows 16,384 at a time. The merged row count is checked against the rows that went in, because a truncated spill file reads back as a valid shorter stream, which is a sorted prefix rather than an error.](/_static/diagrams/sort_run_merge.svg)

## Top-N is not sort-then-slice

The optimizer fuses a `LIMIT` above a `Sort` into `Sort { keys, limit }`, and `parallel_top_n` reduces each morsel to its own top-k before merging the survivors. The input is never concatenated and never fully sorted. The plan shows the fusion, with no `limit` node and the bound pushed into the scan:

```python
ds = bt.from_pydict({"x": [4, 1, 5, 2, 3]})
q = ds.sort("x", descending=True).limit(2)
print(q.to_pydict())
# {'x': [5, 4]}
print(q.explain())
# sort  [top 2 by x]      est≈2  (exact)
# └─ scan  [source 0]     est≈5  (exact)  pushed[top 2 by x desc]
```

![Top-N against sort-then-slice. Sort-then-slice concatenates the morsels into one batch, orders every row and gathers every column, then keeps k, so the relation is copied twice and ordered once to retain a fraction of it. The engine instead reduces each morsel in parallel to its own heap of k, where a row that cannot reach the answer costs one comparison against the current worst; the narrow candidates carry keys only and leave the payload behind; they merge to k rows with ties broken on morsel and row so the survivor matches the stable sort's; and the wide columns are gathered exactly once at the end. The relation is never concatenated and never fully sorted. The heap is used when k is small against the morsel, above which the morsel is sorted and sliced because a linear sort costs no more, and a shared bound can skip a whole morsel whose key range cannot reach the answer, switching itself off after 32 checks that excluded nothing.](/_static/diagrams/topn_heap.svg)

Per morsel, `top_k_indices_of` picks the cheapest shape: a bounded heap when `k` is at most half the morsel, a sort-and-slice for a large `k` over a single text, binary, integer or temporal key, and an O(n) quickselect otherwise. The global merge tie-breaks on each survivor's original `(morsel, row)`, so the kept rows match the stable sort's exactly. On the last full published operator sweep, `sort → LIMIT` ran in 14.1 ms against Polars' 601 ms.

## The gather is the cost

For most sorts the `take` of every column through the permutation costs more than the comparisons. That's why the sample-sort gathers once, and why [`crates/bc-runtime/src/gather/`](https://github.com/stephenoffer/batcher/tree/main/crates/bc-runtime/src/gather) exists. For the byte layouts (`Utf8`, `LargeUtf8`, `Binary`, `LargeBinary`) it does two passes, one to sum lengths into the offset buffer and one to `copy_from_slice` the bytes, instead of Arrow's per-row `extend`. `FixedSizeBinary` fills in parallel, 8x to 22x faster than the single-threaded Arrow `take`. Any other type, a nullable index array, or an offset overflow delegates to Arrow's `take`, so this is a short-circuit and never a second semantics.

## Testing a sort

:::{warning}
Never assert a sort with an order-independent comparison. The differential harness's `assert_same` is a multiset comparison: correct for a group-by and blind to a sort bug. Sort assertions compare sequences, across the cross-product of `{collect, spill, iter_batches, distributed}` and `{nulls, empty, one row, duplicates, -0.0/NaN, descending}` in [`tests/differential/test_diff_operator_matrix.py`](https://github.com/stephenoffer/batcher/blob/main/tests/differential/test_diff_operator_matrix.py).
:::

## Where the code lives

- [`crates/bc-interp/src/ops/mod.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-interp/src/ops/mod.rs): `sort_batch`, `sort_indices`, `sort_indices_of`
- [`crates/bc-interp/src/ops/radix_sort/`](https://github.com/stephenoffer/batcher/tree/main/crates/bc-interp/src/ops/radix_sort): the LSD radix path (`mod.rs`) and the composite packed key (`packed.rs`)
- [`crates/bc-interp/src/ops/byte_sort.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-interp/src/ops/byte_sort.rs): the stable byte-key permutation (text and binary)
- [`crates/bc-runtime/src/byte_key.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-runtime/src/byte_key.rs): the one reading of a byte-key column, shared by the sort and the range partitioner
- [`crates/bc-interp/src/ops/sample_sort/`](https://github.com/stephenoffer/batcher/tree/main/crates/bc-interp/src/ops/sample_sort): the parallel sample-sort
- [`crates/bc-interp/src/ops/run_sort.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-interp/src/ops/run_sort.rs): natural-run detection, shared by the radix and packed-key paths
- [`crates/bc-arrow/src/row_sort.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-arrow/src/row_sort.rs): the row-encoded stable comparison sort
- [`crates/bc-interp/src/ops/external_sort.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-interp/src/ops/external_sort.rs): the spilling k-way merge
- [`crates/bc-runtime/src/gather/`](https://github.com/stephenoffer/batcher/tree/main/crates/bc-runtime/src/gather): the bulk `take`/`concat` fills. `mod.rs` holds the byte layouts and `fixed.rs` the fixed-width ones

## See also

- {doc}`Architecture </architecture/index>`: why an order-defining operator needs its own determinism machinery.
- {doc}`Execution engine </architecture/internals/execution>`: the sequential oracle the five paths must match.
- {doc}`Carbonite </architecture/internals/carbonite>`: the envelope that decides whether the sort goes out of core.
- {doc}`Sorting </user-guide/transform/rows/sorting>`: the API, including `nulls_first` and mixed directions.
- {doc}`Performance </user-guide/operate/tuning/performance>`: why `sort().limit()` is not `sort()` then slice.
- {doc}`Analytics benchmarks </benchmarks/results/analytics>`: the `sort → LIMIT` numbers quoted above.
- {doc}`Sorting at scale </architecture/deep-dives/operators/sort-at-scale>`: which phases grow with the cluster, and how skew is kept from pinning a reducer.
- {doc}`Morsel parallelism </architecture/deep-dives/operators/morsel-parallelism>`: where the ranges get their cores.
- {doc}`Join algorithms </architecture/deep-dives/operators/join-algorithms>`: the other operator that depends on gather cost.
- {doc}`Spilling </architecture/deep-dives/memory/spilling>`: the external merge sort, in its wider context.
