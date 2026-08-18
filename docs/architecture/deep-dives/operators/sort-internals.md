# Sort internals

Sort is the one operator whose *order* is the answer. Every other hash-based operator here
produces an unordered relation, and the tests compare them as multisets, which means a bug in
those paths shows up as a wrong row, loudly. A sort bug shows up as rows in the wrong order,
and an order-independent assertion cannot see it. That asymmetry is why this operator has more
determinism machinery than any other, and why `sort(descending=True)` once silently returned
unsorted data under spill while every gate stayed green.

:::{important}
The engine has five sort paths, and all five must produce the **identical permutation**, not
merely a correctly sorted one. The parallel and spilling paths each sort a *slice* of the input
and concatenate the results, so "sorted" is not enough: two paths that order tied rows
differently produce different relations from the same query.
:::

| Path | Taken when | Code |
|---|---|---|
| LSD radix | a single fixed-width key, a full sort, no `NaN` in the column, floats only up to 2^18 rows | `ops/radix_sort.rs` |
| Composite radix | several keys, all integer or temporal, whose measured value ranges fit one `u64` between them | `ops/radix_sort.rs` |
| Stable string sort | a `Utf8` key | `ops/str_sort.rs` |
| Parallel sample-sort | above 2^17 rows, a full sort, leading key of type float / integer / temporal / string | `ops/sample_sort.rs` |
| External merge sort | the input exceeds the memory envelope | `ops/external_sort.rs` |

```text
   sort_indices(keys, batch)
        │
        ├─ input exceeds the memory envelope? ──────► external merge sort
        │                                               sorted runs → bounded k-way merge
        │
        ├─ > 2^17 rows, full sort, leading key
        │  is int / float / temporal / string? ─────► parallel sample-sort
        │                                               sample → range-partition → sort ranges
        │
        ├─ single fixed-width key, full sort, no NaN,
        │  floats only below 2^18 rows? ────────────► LSD radix sort
        │                                               O(n·w), order-preserving u64 transform
        │
        ├─ several integer / temporal keys whose
        │  measured ranges fit one u64? ────────────► composite radix sort
        │                                               narrow each key to its range, pack, radix
        │
        └─ otherwise ───────────────────────────────► comparison sort
                                                        (the stable string sort for Utf8 keys)

   every path appends the original row index as a final ascending key, so ties resolve
   to input order and the permutation is unique.
```

## The permutation and its tie-break

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
string or boolean key, or a top-N returns `None` and the caller uses the comparison sort. A
multi-key sort declines here too, and is picked up by the composite path below. Floats also decline above 2^18 rows, where the random-scatter key array no
longer fits L2 and the radix loses to the comparison sort outright. Large float sorts arrive
here only per-range (parallel) or per-run (spill), both below that bound.

Nulls are grouped first or last per `nulls_first`, in input order. The sort is stable.

## Path 2: composite radix (several integer or temporal keys)

A multi-key `ORDER BY` had no fast path. `ORDER BY o_orderdate, o_shippriority` fell to the
comparison sort, which encodes every row into Arrow's comparable byte format and then pays
`O(n log n)` memcmps over it. Those two columns hold about 2,400 and 5 distinct values between
them: fifteen bits, against the ninety-six their declared types claim.

`packed_multi_sort_indices` (`ops/radix_sort.rs`) measures each key column's live value range,
gives it exactly `ceil(log2(range))` bits, and packs the whole tuple into one `u64` laid out
most-significant key first. Comparing two packed keys as integers then compares their fields
left to right, stopping at the first difference, which is the definition of lexicographic
order. The sort becomes the same LSD radix path 1 uses, and a fifteen-bit key costs two
counting passes rather than eight.

The narrowing is the idea DuckDB calls compressed materialization
(`src/optimizer/compressed_materialization/compress_order.cpp`), which rewrites a column to
`value - min` at the smallest width its catalog statistics allow. Batcher measures the range on
the rows in hand instead of reading it from a catalog, so it needs no statistics, is exact on
every input, and narrows intermediates that no catalog describes.

Four details carry the correctness:

- **Direction lives in the key, not the sort.** Each column's field holds the order-preserving
  `u64` the single-key radix already uses, which folds `descending` in by inverting. So a sort
  can mix directions per key and the packed integer still sorts ascending.
- **Nulls are encoded inside their field**, not partitioned out, because a multi-key sort's
  nulls are per column and interleaved. A null takes the field's lowest value under
  `nulls_first` and its highest otherwise, and the field is widened by one value to make room,
  only when the column actually has a null.
- **A constant column takes zero bits.** It cannot separate two rows, so it contributes nothing
  and costs nothing.
- **The radix is stable**, so rows equal on every key keep input order, which is the tie-break
  every other path gets from its trailing row-index column.

It declines to the comparison sort on a string, boolean or float key, on fewer than 64 rows, on
more than eight keys, and whenever the measured ranges need more than 64 bits between them. The
decline is cheap by construction: a range measured over a *prefix* of the rows can only widen,
so 4,096 rows are enough to *reject* a key that will not fit, and the exact scan runs only for a
key that might. Rejecting is always legal, which is what lets the probe be one-sided.

Measured on 8 M rows, best of three interleaved runs against the path it replaces:

| Shape | Before | After | |
|---|---|---|---|
| `ORDER BY <date>, <priority>` (12,000 distinct pairs) | 1,878 ms | 625 ms | **3.0x** |
| `ORDER BY <int>, <int>` (3 M distinct pairs) | 112 ms | 59 ms | **1.9x** |
| `ORDER BY <int> DESC, <int>` | 90 ms | 60 ms | **1.5x** |
| `ORDER BY <int>, <int>, <int>` (narrow) | 133 ms | 71 ms | **1.9x** |
| `ORDER BY <full-width int>, <full-width int>` (declines) | 66 ms | 61 ms | 1.1x |
| `ORDER BY <string>, <int>` (declines) | 199 ms | 200 ms | 1.0x |

Those figures isolate the packing. On the committed benchmark case the two changes on this page
compound: `op-sort-multikey-narrow` (`ORDER BY l_shipdate, l_suppkey` over 6M `lineitem` rows)
goes from **1,590 ms to 52 ms**, which is 37.1x slower than DuckDB to 1.28x. The packing alone
accounts for about 3x of that; the rest is the sample-sort note under path 4.

## Path 3: stable string sort

`ops/str_sort.rs`. Strings had no radix path, so a string `ORDER BY` was the one sort whose tie
order was nondeterministic, and that nondeterminism is precisely what blocked the parallel
sample-sort on a string key, because a range's tie order could not be made to agree with the
whole-array sort's.

This module supplies the missing guarantee: nulls grouped by `nulls_first` in input order,
non-null rows sorted byte-lexicographically (the ordering Arrow itself uses for `Utf8`), and
descending inverting only the key comparison, never the tie-break. Equal keys always keep input
order, so the serial oracle and the per-range sorts produce the identical relation.

## Path 4: parallel sample-sort

`ops/sample_sort.rs`, above 2^17 rows, for a full sort with a float, integer, temporal, or
string leading key.

Sample ~8,192 rows to estimate quantile boundaries, range-partition the rows by the leading key,
and sort each range in parallel. The ranges are globally ordered relative to each other, so the
sorted relation is the ranges in key order: no final merge, and no concatenation, because
the executor consumes a `Vec<RecordBatch>` anyway.

```text
   sample ~8,192 rows ──► quantile boundaries   b0 < b1 < b2
                                │
                                ▼
   route each row to a range by its leading key   (row INDICES only, no payload copy)
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
   the ranges, in key order, ARE the sorted relation: no merge, no concat
```

The payload is gathered exactly once, and that is the whole point of the design. Materializing
the ranges up front, and again to sort them, and a third time to concatenate, copied every
column three times. On a 5M-row, 6-column sort that was two thirds of the work.

Multi-key sorts work: rows bucket by the leading key (equal leading keys never span a boundary),
then each range sorts by the full key list, so a plain concatenation in leading-key order is the
globally sorted multi-key relation.

This is the single-node form of the distributed range sort, and it comes from the same
`bc_runtime::shuffle` range partitioners: one implementation, two scales.

:::{note}
A temporal leading key routes as `i64`, and until recently did not. The routing tested
`DataType::is_integer()`, which is false for `Date32`, `Date64`, `Timestamp`, `Time32/64` and
`Duration` — so every one of them fell through and `ORDER BY <date>` ran **serially**, on
however many cores the box had. On 6M rows a date sort with 2,500 distinct values took 643 ms,
against 38 ms for the same sort on an `Int64` column holding *more* distinct values. Those types
are physically signed integers whose numeric order is the type's order, in every unit and with
or without a time zone, so they take the same routing. Two stay out. `Interval` is not one
integer: `MonthDayNano` is three fields in 128 bits and none orders it. `Time32` is one integer
that Arrow will not widen, so admitting it would raise rather than sort; the routing also
declines on a failed cast, which keeps that class of mistake to a missed optimization.
:::

## Path 5: external merge sort

`ops/external_sort.rs`, when the input exceeds the memory envelope. Sort each input morsel into a
run and spill it (dropping the input batch as you go), then merge the runs with a bounded-fan-in
streaming k-way merge over a `BinaryHeap` of encoded rows.

Peak memory is O(`sort_merge_fanin` morsels) regardless of input size: only one batch per run in
the active merge group is resident, and the output streams back to disk between passes. The
result equals a single in-memory `sort_batch` over the whole input. Spill files are Arrow IPC
through the same `DiskSpillStore` the aggregate uses.

## Top-N is not sort-then-slice

A `LIMIT` above a `Sort` is fused by the optimizer into `Sort { keys, limit }`, and `parallel_top_n`
reduces each morsel to its own local top-k before merging the narrow set of survivors. The whole
input is never concatenated and never fully sorted.

Per morsel, `top_k_indices` picks between two shapes. A single `Utf8` key uses the stable string
permutation builder and a single integer or temporal key uses the radix, because both are linear,
so sorting the whole morsel costs no more than selecting k from it. Every other key, including a
single float key, falls through to an O(n) quickselect over a total-order row comparator that walks
each `ORDER BY` key in turn and then the row index. A float key used to take the radix full sort
too, and that was the wrong trade: the float radix runs eight LSD passes scattering by a random key
byte, so sorting a 13k-row morsel to keep 100 rows costs roughly eight times an O(n) selection and
thrashes cache. Routing float to the quickselect moved `ORDER BY <f64> DESC LIMIT 100` over 6M rows
from 3.03x DuckDB to 1.57x.

Quickselect is unstable, so the order in which a morsel hands back its candidates is arbitrary. The
global merge therefore tie-breaks on the survivor's original `(morsel, row)` rather than on its
position in the flattened candidate array. Those two agree only when every morsel returns rows in
ascending-row order, which the radix full sort happened to do and the quickselect does not, and the
gap between them was a live wrong-answer bug: a different tied row survived at the same rank than
the stable oracle keeps, visible only with real key ties, since a distinct second sort key removes
them. `parallel_top_n_float_key_matches_eager` pins the fixed behavior across `-0.0`/`0.0`, `NaN`,
heavy ties, and every ascending/descending by nulls-first combination.

The fusion is worth the machinery. On the operator sweep `sort → LIMIT` sits at parity with DuckDB
and runs 33x faster than Polars, which sorts the whole relation and then slices it.

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

The third line is the composite shape. With a text or binary leading key it takes the comparison
sort;
swap `g` for a second integer or date column and the same call takes the composite radix, with
no change to the query and none to the result:

```python
import batcher as bt

orders = bt.from_pydict(
    {
        "orderdate": [19_950_301, 19_950_302, 19_950_301, 19_950_302],
        "shippriority": [0, 1, 1, 0],
        "revenue": [10.0, 20.0, 30.0, 40.0],
    }
)
print(orders.sort("orderdate", "shippriority", descending=[False, True]).to_pydict())
```

```text
{'orderdate': [19950301, 19950301, 19950302, 19950302], 'shippriority': [1, 0, 1, 0], 'revenue': [30.0, 10.0, 20.0, 40.0]}
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
- `crates/bc-interp/src/ops/radix_sort.rs`: the LSD radix path, and the composite key packed from measured ranges
- `crates/bc-interp/src/ops/str_sort.rs`: the stable string permutation
- `crates/bc-interp/src/ops/sample_sort.rs`: the parallel sample-sort
- `crates/bc-interp/src/ops/external_sort.rs`: the spilling k-way merge
- `crates/bc-runtime/src/gather.rs`: the string `take` fast path

## See also

- {doc}`Architecture </architecture/index>`: why an order-defining operator needs its own determinism machinery.
- {doc}`Execution engine </architecture/internals/execution>`: the sequential oracle the five paths must match.
- {doc}`Carbonite </architecture/internals/carbonite>`: the envelope that decides whether the sort goes out of core.
- {doc}`Sorting </user-guide/transform/rows/sorting>`: the API, including `nulls_first` and mixed directions.
- {doc}`Performance </user-guide/operate/tuning/performance>`: why `sort().limit()` is not `sort()` then slice.
- {doc}`Analytics benchmarks </benchmarks/results/analytics>`: the `sort → LIMIT` numbers quoted above.
- {doc}`Morsel parallelism </architecture/deep-dives/operators/morsel-parallelism>`: where the ranges get their cores.
- {doc}`Join algorithms </architecture/deep-dives/operators/join-algorithms>`: the other operator that depends on gather cost.
- {doc}`Spilling </architecture/deep-dives/memory/spilling>`: the external merge sort, in its wider context.
