# Sort internals

Sort is the one operator whose *order* is the answer. Every other hash-based operator here
produces an unordered relation, and the tests compare them as multisets, which means a bug in
those paths shows up as a wrong row, loudly. A sort bug shows up as rows in the wrong order,
and an order-independent assertion cannot see it. That asymmetry is why this operator has more
determinism machinery than any other, and why `sort(descending=True)` once silently returned
unsorted data under spill while every gate stayed green.

:::{important}
The engine has four sort paths, and all four must produce the **identical permutation**, not
merely a correctly sorted one. The parallel and spilling paths each sort a *slice* of the input
and concatenate the results, so "sorted" is not enough: two paths that order tied rows
differently produce different relations from the same query.
:::

| Path | Taken when | Code |
|---|---|---|
| LSD radix | a single fixed-width key, a full sort, no `NaN` in the column, floats only up to 2^18 rows | `ops/radix_sort.rs` |
| Stable string sort | a `Utf8` key | `ops/str_sort.rs` |
| Parallel sample-sort | above 2^17 rows, a full sort, leading key of type float / integer / string | `ops/sample_sort.rs` |
| External merge sort | the input exceeds the memory envelope | `ops/external_sort.rs` |

```text
   sort_indices(keys, batch)
        │
        ├─ input exceeds the memory envelope? ──────► external merge sort
        │                                               sorted runs → bounded k-way merge
        │
        ├─ > 2^17 rows, full sort, leading key
        │  is int / float / string? ────────────────► parallel sample-sort
        │                                               sample → range-partition → sort ranges
        │
        ├─ single fixed-width key, full sort, no NaN,
        │  floats only below 2^18 rows? ────────────► LSD radix sort
        │                                               O(n·w), order-preserving u64 transform
        │
        └─ otherwise ───────────────────────────────► comparison sort
                                                        (the stable string sort for Utf8 keys)

   every path appends the original row index as a final ascending key, so ties resolve
   to input order and the permutation is unique.
```

## The permutation, and the tie-break that makes it unique

`sort_indices` (`crates/bc-interp/src/ops/mod.rs`) evaluates the key expressions and builds a
`UInt32Array` permutation. `sort_batch` then `take`s the whole batch through it.

Arrow's comparison sorts are not stable. `lexsort_to_indices` and `sort_to_indices` leave rows
with equal keys in an arbitrary, input-size-dependent order. For a single-node engine that is
merely surprising; for this one it is a correctness bug, because a range of the input sorted on
one thread and the whole input sorted on another would order tied rows differently, and `seq ==
par` would fail bit-for-bit.

So every path appends the **original row index as a final ascending key**. Ties then resolve to
input order, a deterministic total order and exactly what a stable sort yields. The slice a
parallel range sorts is always gathered in ascending original-row order, so a slice-local `0..n`
index preserves the input's relative order of tied rows within it.

## Path 1: radix (single fixed-width key, full sort)

`ops/radix_sort.rs`. An LSD radix sort over an order-preserving `u64` transform of the key:
sign-flipped for signed integers, bit-inverted for descending, an IEEE-order transform matching
Arrow's `total_cmp` for floats. O(n·w) in the key width against the comparison sort's O(n log n),
producing the identical relation.

It declines rather than risks: a column containing a `NaN` (no single numeric position), a
string or boolean key, a multi-key sort, or a top-N returns `None` and the caller uses the
comparison sort. Floats also decline above 2^18 rows, where the random-scatter key array no
longer fits L2 and the radix loses to the comparison sort outright. Large float sorts arrive
here only per-range (parallel) or per-run (spill), both below that bound.

Nulls are grouped first or last per `nulls_first`, in input order. The sort is stable.

## Path 2: stable string sort

`ops/str_sort.rs`. Strings had no radix path, so a string `ORDER BY` was the one sort whose tie
order was nondeterministic, and that nondeterminism is precisely what blocked the parallel
sample-sort on a string key, because a range's tie order could not be made to agree with the
whole-array sort's.

This module supplies the missing guarantee: nulls grouped by `nulls_first` in input order,
non-null rows sorted byte-lexicographically (the ordering Arrow itself uses for `Utf8`), and
descending inverting only the key comparison, never the tie-break. Equal keys always keep input
order, so the serial oracle and the per-range sorts produce the identical relation.

## Path 3: parallel sample-sort

`ops/sample_sort.rs`, above 2^17 rows, for a full sort with a float, integer, or string leading
key.

Sample ~8,192 rows to estimate quantile boundaries, range-partition the rows by the leading key,
and sort each range in parallel. The ranges are globally ordered relative to each other, so the
sorted relation is just the ranges in key order: no final merge, and no concatenation, because
the executor consumes a `Vec<RecordBatch>` anyway.

```text
   sample ~8,192 rows ──► quantile boundaries   b0 < b1 < b2
                                │
                                ▼
   route each row to a range by its leading key   (row INDICES only — no payload copy)
   ┌──────────┬──────────┬──────────┬──────────┐
   │ range 0  │ range 1  │ range 2  │ range 3  │  globally ordered relative to each other
   └────┬─────┴────┬─────┴────┬─────┴────┬─────┘
        │          │          │          │
     sort keys  sort keys  sort keys  sort keys   in parallel, on a cheap gather of the
        │          │          │          │        key columns alone
        ▼          ▼          ▼          ▼
      compose that permutation with the range's row indices
        │          │          │          │
        └──────────┴────┬─────┴──────────┘
                        ▼
         gather every column ONCE, per range
                        │
                        ▼
   the ranges, in key order, ARE the sorted relation — no merge, no concat
```

The payload is gathered exactly once, and that is the whole point of the design. Materializing
the ranges up front, and again to sort them, and a third time to concatenate, copied every
column three times. On a 5M-row, 6-column sort that was two thirds of the work.

Multi-key sorts work: rows bucket by the leading key (equal leading keys never span a boundary),
then each range sorts by the full key list, so a plain concatenation in leading-key order is the
globally sorted multi-key relation.

This is the single-node form of the distributed range sort, and it comes from the same
`bc_runtime::shuffle` range partitioners: one implementation, two scales.

## Path 4: external merge sort

`ops/external_sort.rs`, when the input exceeds the memory envelope. Sort each input morsel into a
run and spill it (dropping the input batch as you go), then merge the runs with a bounded-fan-in
streaming k-way merge over a `BinaryHeap` of encoded rows.

Peak memory is O(`sort_merge_fanin` morsels) regardless of input size: only one batch per run in
the active merge group is resident, and the output streams back to disk between passes. The
result equals a single in-memory `sort_batch` over the whole input. Spill files are Arrow IPC
through the same `DiskSpillStore` the aggregate uses.

## Top-N is not sort-then-slice

A `LIMIT` above a `Sort` is fused by the optimizer into `Sort { keys, limit }`, and the engine
runs a *partial* sort, O(n log k) rather than O(n log n). That partial sort is unstable, so
which tied rows survive is arbitrary and input-size-dependent, which would make single-node and
the distributed range sort (whose per-bucket reduce runs the same top-N over a differently sized
slice) disagree on ties. The row-index tie-break key fixes it: the top-N is deterministic and
identical to a stable sort-then-slice, for one extra unique key in the same partial sort.

This fusion is where the engine's biggest sort win comes from: `sort → LIMIT` runs in 14.1 ms
against DuckDB's 13.3 ms and Polars' 600.7 ms, because Polars sorts the whole relation and then slices.

## The gather is the cost

For most sorts the comparison work is not the bottleneck; the `take` of every column through the
permutation is. That is why the sample-sort works so hard to gather once.

It is also why `crates/bc-runtime/src/gather.rs` exists. Arrow's `take` on `Utf8`/`LargeUtf8` is
far slower than the memory it moves: it drives `MutableArrayData::extend` once per row, paying a
call and bounds checks to copy a handful of bytes. On a 5M-row sort, adding one string column cost
~52 ms, an order of magnitude more than the ~50 MB of characters involved. The fast path does two
passes: one to sum the gathered lengths into the offset buffer, one to `copy_from_slice` the bytes.
Every other type, a nullable index array, or an offset overflow delegates to Arrow's `take`, so it
is a pure short-circuit and never a second semantics.

## Using it

```python
import batcher as bt

ds = bt.from_pydict({"x": [5, 1, 4, 2, None], "g": ["a", "b", "a", "b", "c"]})

print(ds.sort("x").to_pydict())                                  # nulls last by default
print(ds.sort("x", descending=True).limit(2).to_pydict())        # fused into a top-N
print(ds.sort("g", "x", descending=[False, True]).to_pydict())   # multi-key, mixed direction
print(ds.sort("x", descending=True).limit(2).explain())
```

```text
{'x': [1, 2, 4, 5, None], 'g': ['b', 'b', 'a', 'a', 'c']}
{'x': [5, 4], 'g': ['a', 'a']}
{'x': [5, 4, 2, 1, None], 'g': ['a', 'a', 'b', 'b', 'c']}
```

:::{dropdown} The explained plan, with the fusion visible
```text
sort                            est≈2 (exact)
  scan                          est≈5 (exact)
```

There is no `limit` node above the sort. The sort node absorbed it, and runs a partial sort
instead of a full one.
:::

## The rule for testing this

:::{warning}
Never assert a sort with an order-independent comparison. The differential harness's
`assert_same` is a multiset comparison: correct for a group-by, and completely blind to a sort
bug. Sort assertions compare sequences, across the cross-product of `{collect, spill,
iter_batches, distributed}` and `{nulls, empty, one row, duplicates, -0.0/NaN, descending}`.
That cross-product is `tests/differential/test_diff_operator_matrix.py`, and it exists because
the shape that broke, `descending=True` under spill, was never the shape anyone was thinking
about.
:::

## Where the code lives

- `crates/bc-interp/src/ops/mod.rs`: `sort_batch`, `sort_indices`, `sort_indices_of`
- `crates/bc-interp/src/ops/radix_sort.rs`: the LSD radix path
- `crates/bc-interp/src/ops/str_sort.rs`: the stable string permutation
- `crates/bc-interp/src/ops/sample_sort.rs`: the parallel sample-sort
- `crates/bc-interp/src/ops/external_sort.rs`: the spilling k-way merge
- `crates/bc-runtime/src/gather.rs`: the string `take` fast path

## See also

:::{seealso}
- [Architecture](../architecture/index.md): why an order-defining operator needs its own determinism machinery
- [Execution engine](../internals/execution.md): the sequential oracle the four paths must match
- [Carbonite](../internals/carbonite.md): the envelope that decides whether the sort goes out of core
- [Sorting](../user-guide/sorting.md): the API, including `nulls_first` and mixed directions
- [Performance](../user-guide/performance.md): why `sort().limit()` is not `sort()` then slice
- [Analytics benchmarks](../benchmarks/analytics.md): the `sort → LIMIT` numbers quoted above
- [Morsel parallelism](morsel-parallelism.md): where the ranges get their cores
- [Join algorithms](join-algorithms.md): the other operator that depends on gather cost
- [Spilling](spilling.md): the external merge sort, in its wider context
:::
